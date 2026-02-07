# Copyright 2025 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import re
import signal
import traceback

from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Thread
from typing import Any

from hivemind.utils import get_dht_time
from hivemind.utils.logging import get_logger

from node0.security.validation import WorkerMetricsV1
from node0.utils.flops import (
    get_num_flop_per_token_bwd,
    get_num_flop_per_token_fwd,
    get_num_params,
)


logger = get_logger(__name__)


class CustomFileFormatter(logging.Formatter):
    """
    A formatter that logs time in UTC.
    """

    converter = lambda *args: datetime.now(timezone.utc).timetuple()

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if record.levelno > logging.INFO:
            if not hasattr(record, "caller"):
                record.caller = f"{record.name}.{record.funcName}:{record.lineno}"
            record.caller_block = f" [{record.caller}]"
        else:
            record.caller_block = ""

        return super().format(record)


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        log_entry = self.format(record)

        # Traceback object can't be stored in mp.Queue, converting to str
        if record.exc_info is not None:
            try:
                new_exc_info = (
                    record.exc_info[0],
                    record.exc_info[1],
                    "".join(traceback.format_tb(record.exc_info[2])),
                )
                record.exc_info = new_exc_info  # using str instead of TracebackType, but it still works
            except Exception:
                record.exc_info = None

        self.log_queue.put({"text": log_entry, "record": record})


class BaseMonitor(Thread):
    def __init__(
        self,
        log_file: str | None = None,
        save_clean_logs: bool = False,
    ):
        super().__init__(daemon=True)

        # Create a log handler
        self._log_queue = mp.Queue()
        self.queue_handler = QueueHandler(self._log_queue)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.queue_handler.setFormatter(formatter)

        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

            if save_clean_logs:
                file_formatter = formatter
            else:
                file_formatter = CustomFileFormatter(
                    fmt="{asctime}.{msecs:03.0f} [{levelname}]{caller_block} {message}",
                    style="{",
                    datefmt="%b %d %H:%M:%S",
                )
            self.file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 10, backupCount=20)
            self.file_handler.setFormatter(file_formatter)
        else:
            self.file_handler = None


class MonitorWorker(BaseMonitor):
    def __init__(
        self,
        peer_id: str = "",
        local_public_key: bytes | None = None,
        stats_report_interval: int = 60,
        experiment_prefix: str = "",
        stage: str = "",
        authorizer: Any | None = None,  # can't import real type because of circular import
        save_clean_logs: bool = False,
        log_file: str | None = None,
        terminate_AR_fail: bool = True,
        terminate_AR_limit: int = 2,
        terminate_AR_state_limit: int = 2,
        terminate_load_state_limit: int = 1,
    ):
        super().__init__(log_file=log_file, save_clean_logs=save_clean_logs)
        self.dht = None
        self.authorizer = authorizer
        self.experiment_prefix = experiment_prefix
        self.stage = stage
        self.peer_id = peer_id
        self.local_public_key = local_public_key
        self.stats_report_interval = stats_report_interval
        self.internal_report_interval = 3  # number of reports
        self.scaling = float(10**12)
        self.last_reauthorized = datetime.now(timezone.utc)
        self.last_active = None
        self.active_period_timeout = None
        self.should_reauthorize = False
        self.AR_fail_cnt = 0
        self.AR_state_fail_cnt = 0
        self.load_state_fail_cnt = 0
        self.terminate_AR_fail = terminate_AR_fail
        self.terminate_AR_limit = terminate_AR_limit
        self.terminate_AR_state_limit = terminate_AR_state_limit
        self.terminate_load_state_limit = terminate_load_state_limit
        self.port_check_interval = 30

        # Log patterns
        self.processed_pattern = r".*Processed (\d+) batches.*"
        self.forward_pattern = r"([\d-]+ [\d:,]+).*?_forward:.*?(\d+)\s+examples"
        self.backward_pattern = r"([\d-]+ [\d:,]+).*?_backward:.*?(\d+)\s+examples"
        self.failed_find_group_pattern = r"Averaging step failed: could not find a group"
        self.AR_gradients_timeout_pattern = r"Averaging gradients failed with TimeoutError"
        self.AR_state_timeout_pattern = r"Averaging failed with <class 'TimeoutError'>"
        self.AR_gradients_success_pattern = r"Averaged gradients with (\d+) peers"
        self.AR_state_success_pattern = r"Averaged parameters with (\d+) peers"
        self.load_state_peers_pattern = r".*Failed to load state from.*"
        self.error_in_grads = r".*Encountered incorrect value in grads.*"
        self.authorization_fail_pattern = r".*Authorization failed:.*"
        self.cuda_error_pattern = r".*Caught CUDA error.*"
        self.start_pattern = r".*Server started with.*"
        self.accumulate_pattern = r".*accumulated (-?\d+) samples for epoch.*"
        self.connection_error_pattern = r".*Connection refused.*"
        self.port_error_pattern = r".*Port test failed.*"

        # Values for computing flops
        self.num_flop_per_token_fwd = 0
        self.num_flop_per_token_bwd = 0
        self.seq_len = 0
        self.active_time = []  # seconds
        self.num_flop = []
        self.n_reports = 0

    def connect_dht(self, dht):
        """Connect to existing DHT."""
        self.dht = dht

    def add_auth_info(
        self,
        authorizer: Any,
        peer_id: str,
        stage: str,
        local_public_key: bytes,
    ):
        """Update monitor with authorization details."""
        self.authorizer = authorizer
        self.peer_id = peer_id
        self.stage = stage
        self.local_public_key = local_public_key
        self.monitor_public_key = authorizer.monitor_public_key

    def monitor_callback(self, model, model_conf, active_period_timeout):
        """Get required values from model config."""
        num_params = get_num_params(model, exclude_embedding=True)
        self.num_flop_per_token_fwd = get_num_flop_per_token_fwd(num_params, model_conf, model_conf.max_seq_len)
        self.num_flop_per_token_bwd = get_num_flop_per_token_bwd(num_params, model_conf, model_conf.max_seq_len)
        self.seq_len = int(model_conf.max_seq_len)
        self.active_period_timeout = timedelta(seconds=active_period_timeout)

    def report(self, current_time):
        """Save report to DHT."""
        if self.dht is None:
            logger.error("DHT is not connected to Monitor Worker, can't report")
            return

        if self.local_public_key is None:
            logger.error("local_public_key is None, can't report")
            return

        # Don't report if no work is done
        if len(self.active_time) == 0:
            self.n_reports = 0
            return

        # Prepare metrics
        metrics = WorkerMetricsV1(
            peer_id=self.peer_id,
            num_flop=sum(self.num_flop),
            active_time=sum(self.active_time),
        )

        # Save to DHT
        self.dht.store(
            key=f"{self.experiment_prefix}_{self.stage}_worker_metrics",
            subkey=self.local_public_key,
            value=metrics.dict(),
            expiration_time=current_time + self.internal_report_interval * self.stats_report_interval,
            return_future=True,
        )

        # Clean
        self.flops_fwd = []
        self.flops_bwd = []
        self.num_flop = []
        self.active_time = []
        self.n_reports = 0

    def _run_async_port_check(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._port_check())

    async def _port_check(self):
        while True:
            if self.dht:
                await self.get_port_info()
            await asyncio.sleep(self.port_check_interval)

    async def get_port_info(self):
        """Check if DHT contains record about port reachability"""
        try:
            response = await self.dht.get(
                key=f"{self.experiment_prefix}_{self.peer_id}_worker_ports", latest=True, return_future=True
            )

            if not response:
                return

            response = list(response.value.items())
            assert len(response) == 1
            subkey = response[0][0]

            if subkey == f"[owner:{self.monitor_public_key}]".encode():
                logger.error("Port test failed. Make sure your announced port is open")

        except Exception as e:
            logger.error(f"Port test failed: {e}")

    def run(self):
        """Run monitor in background to report worker's contribution."""
        async_thread = Thread(target=self._run_async_port_check, daemon=True)
        async_thread.start()

        while True:
            try:
                log_dict = self._log_queue.get(timeout=1)  # Wait for a log entry.
                log_entry = log_dict["text"]
                record = log_dict["record"]

                if self.file_handler:
                    self.file_handler.handle(record)

                # Check activity
                if (
                    self.last_active is not None
                    and self.active_period_timeout is not None
                    and (datetime.now(timezone.utc) - self.last_active) > self.active_period_timeout
                ):
                    logger.error("An unknown server error occurred. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)

                accumulate_match = re.search(self.accumulate_pattern, log_entry)
                if accumulate_match:
                    self.last_active = datetime.now(timezone.utc)
                    continue

                connection_error_match = re.search(self.connection_error_pattern, log_entry)
                if connection_error_match:
                    logger.error("An unknown server error occurred. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)

                authorization_fail_match = re.search(self.authorization_fail_pattern, log_entry)
                if authorization_fail_match:
                    logger.error("Failed to authorize. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)

                port_error_match = re.search(self.port_error_pattern, log_entry)
                if port_error_match:
                    logger.error("Failed to connect to the port. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)

                cuda_error_match = re.search(self.cuda_error_pattern, log_entry)
                if cuda_error_match:
                    logger.error("Caught cuda error. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)

                load_state_peer_fail = re.search(self.load_state_peers_pattern, log_entry, re.DOTALL)
                if load_state_peer_fail:
                    self.load_state_fail_cnt += 1
                    if self.load_state_fail_cnt >= self.terminate_load_state_limit:
                        logger.error("Failed to load state from peers. Exiting run.")
                        os.killpg(os.getpgrp(), signal.SIGTERM)

                error_in_grads = re.search(self.error_in_grads, log_entry)
                if error_in_grads:
                    logger.error("Caught invalid value in gradients. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)

                AR_state_success = re.search(self.AR_state_success_pattern, log_entry)
                if AR_state_success:
                    self.AR_state_fail_cnt = 0
                    self.AR_fail_cnt = 0
                    continue

                AR_gradients_success = re.search(self.AR_gradients_success_pattern, log_entry)
                if AR_gradients_success:
                    self.AR_fail_cnt = 0
                    continue

                # Test for AR reduce fail and terminate peer
                failed_find_group = re.search(self.failed_find_group_pattern, log_entry)
                AR_gradient_timeout = re.search(self.AR_gradients_timeout_pattern, log_entry)
                AR_state_timeout = re.search(self.AR_state_timeout_pattern, log_entry)

                if AR_state_timeout:
                    self.AR_state_fail_cnt += 1
                    if (self.AR_state_fail_cnt >= self.terminate_AR_state_limit) and self.terminate_AR_fail:
                        logger.error("Too many failed state all-reduce attempts. Exiting run.")
                        os.killpg(os.getpgrp(), signal.SIGTERM)

                if AR_gradient_timeout or AR_state_timeout or failed_find_group:
                    self.AR_fail_cnt += 1
                    if (self.AR_fail_cnt >= self.terminate_AR_limit) and self.terminate_AR_fail:
                        logger.error("Too many failed all-reduce attempts. Exiting run.")
                        os.killpg(os.getpgrp(), signal.SIGTERM)

                # Start of report
                processed_match = re.search(self.processed_pattern, log_entry)
                if processed_match:
                    if self.n_reports >= self.internal_report_interval:
                        # Record metrics
                        current_time = get_dht_time()
                        self.report(current_time)

                    self.n_reports += 1

                    # Only count active time when some batches were processed
                    try:
                        processed_batches = int(processed_match.group(1))
                        if processed_batches > 0:
                            self.active_time.append(float(self.stats_report_interval))
                    except Exception as e:
                        logger.error(f"Can't report time: {e}")

                    continue

                # Forward flops
                forward_match = re.search(self.forward_pattern, log_entry)
                if forward_match:
                    try:
                        forward_examples = max(0, int(forward_match.group(2)))
                        num_flop = forward_examples * self.seq_len * self.num_flop_per_token_fwd / self.scaling
                        self.num_flop.append(num_flop)
                    except Exception as e:
                        logger.error(f"Can't report forward flop: {e}")
                    continue

                # Backward flops
                backward_match = re.search(self.backward_pattern, log_entry)
                if backward_match:
                    try:
                        backward_examples = max(0, int(backward_match.group(2)))
                        num_flop = backward_examples * self.seq_len * self.num_flop_per_token_bwd / self.scaling
                        self.num_flop.append(num_flop)
                    except Exception as e:
                        logger.error(f"Can't report backward flop: {e}")
                    continue

                start_match = re.search(self.start_pattern, log_entry)
                if start_match:
                    self.should_reauthorize = True
                    self.last_active = datetime.now(timezone.utc)
                    continue

                if self.authorizer and self.authorizer.reachable == "unknown" and self.should_reauthorize:
                    # Check if need to re-authorize:
                    current_time = datetime.now(timezone.utc)
                    if (current_time - self.last_reauthorized) > timedelta(minutes=2):
                        try:
                            self.authorizer.join_experiment(n_retries=1)
                            self.last_reauthorized = current_time
                        except Exception as e:
                            logger.error(f"Authorization failed: {e}. Exiting run.")
                            os.killpg(os.getpgrp(), signal.SIGTERM)

            except queue.Empty:
                continue
