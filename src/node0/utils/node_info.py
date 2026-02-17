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

import re
import time

from collections.abc import Callable
from typing import Any

import psutil
import speedtest
import torch

from hivemind.utils.logging import get_logger
from pydantic import BaseModel


logger = get_logger(__name__)


SPEEDTEST_SERVERS = [
    {
        "url": "http://speedtest-wa.waveip.org:8080/speedtest/upload.php",
        "lat": "47.6062",
        "lon": "-122.3321",
        "name": "Seattle, WA",
        "country": "United States",
        "cc": "US",
        "sponsor": "Wave",
        "id": "60635",
        "host": "speedtest-wa.waveip.org:8080",
    },
    {
        "url": "http://speedtest.sea1.nitelusa.net:8080/speedtest/upload.php",
        "lat": "47.6062",
        "lon": "-122.3321",
        "name": "Seattle, WA",
        "country": "United States",
        "cc": "US",
        "sponsor": "Nitel",
        "id": "12192",
        "host": "speedtest.sea1.nitelusa.net:8080",
    },
    {
        "url": "http://us-sea02.speed.misaka.one:8080/speedtest/upload.php",
        "lat": "47.6062",
        "lon": "-122.3321",
        "name": "Seattle, WA",
        "country": "United States",
        "cc": "US",
        "sponsor": "Misaka Network, Inc.",
        "id": "50679",
        "host": "us-sea02.speed.misaka.one:8080",
    },
    {
        "url": "http://sea-speedtest.net.sangoma.net:8080/speedtest/upload.php",
        "lat": "47.6062",
        "lon": "-122.3321",
        "name": "Seattle, WA",
        "country": "United States",
        "cc": "US",
        "sponsor": "Sangoma",
        "id": "63939",
        "host": "sea-speedtest.net.sangoma.net:8080",
    },
    {
        "url": "http://wa01svp-speed11.svc.tds.net:8080/speedtest/upload.php",
        "lat": "47.6062",
        "lon": "-122.3321",
        "name": "Seattle, WA",
        "country": "United States",
        "cc": "US",
        "sponsor": "TDS Telecom",
        "id": "70208",
        "host": "wa01svp-speed11.svc.tds.net:8080",
    },
    {
        "url": "http://speedtest-jp.tgb-host.com:8080/speedtest/upload.php",
        "lat": "35.6833",
        "lon": "139.6833",
        "name": "Tokyo",
        "country": "Japan",
        "cc": "JP",
        "sponsor": "7 BULL",
        "id": "65101",
        "host": "speedtest-jp.tgb-host.com:8080",
    },
    {
        "url": "http://speedtest.jp230.hnd.jp.ctcsci.com:8080/speedtest/upload.php",
        "lat": "35.6833",
        "lon": "139.6833",
        "name": "Tokyo",
        "country": "Japan",
        "cc": "JP",
        "sponsor": "CTCSCI TECH LTD",
        "id": "62217",
        "host": "speedtest.jp230.hnd.jp.ctcsci.com:8080",
    },
    {
        "url": "http://speedtest.3s-labo.com:8080/speedtest/upload.php",
        "lat": "35.6074",
        "lon": "140.1065",
        "name": "Chiba",
        "country": "Japan",
        "cc": "JP",
        "sponsor": "3s-labo",
        "id": "70451",
        "host": "speedtest.3s-labo.com:8080",
    },
    {
        "url": "http://sto-ste-speedtest1.bahnhof.net:8080/speedtest/upload.php",
        "lat": "59.3294",
        "lon": "18.0686",
        "name": "Stockholm",
        "country": "Sweden",
        "cc": "SE",
        "sponsor": "Bahnhof AB",
        "id": "34024",
        "host": "sto-ste-speedtest1.bahnhof.net:8080",
    },
    {
        "url": "http://fd.sunet.se:8080/speedtest/upload.php",
        "lat": "59.3294",
        "lon": "18.0686",
        "name": "Stockholm",
        "country": "Sweden",
        "cc": "SE",
        "sponsor": "SUNET",
        "id": "26852",
        "host": "fd.sunet.se:8080",
    },
    {
        "url": "http://speedtest-sth.netatonce.net:8080/speedtest/upload.php",
        "lat": "59.3294",
        "lon": "18.0686",
        "name": "Stockholm",
        "country": "Sweden",
        "cc": "SE",
        "sponsor": "Net at Once Sweden AB",
        "id": "63781",
        "host": "speedtest-sth.netatonce.net:8080",
    },
    {
        "url": "http://se-speedt02.hy.nis.telia.net:8080/speedtest/upload.php",
        "lat": "59.3294",
        "lon": "18.0686",
        "name": "Stockholm",
        "country": "Sweden",
        "cc": "SE",
        "sponsor": "Telia Sweden AB",
        "id": "45936",
        "host": "se-speedt02.hy.nis.telia.net:8080",
    },
    {
        "url": "http://speedtest-sth.84grams.net:8080/speedtest/upload.php",
        "lat": "59.3294",
        "lon": "18.0686",
        "name": "Stockholm",
        "country": "Sweden",
        "cc": "SE",
        "sponsor": "84 Grams AB",
        "id": "53521",
        "host": "speedtest-sth.84grams.net:8080",
    },
]


class NodeInfo(BaseModel):
    device_name: str
    gpu_memory: float  # GB
    ram: float  # GB
    download_speed: float | None
    upload_speed: float | None
    latency: float | None


class NonRetriableError(Exception):
    pass


class RetriableError(Exception):
    pass


class NotInAllowlistError(NonRetriableError):
    pass


class BadRequestError(NonRetriableError):
    pass


class IntegrityError(NonRetriableError):
    pass


class ServerUnavailableError(RetriableError):
    pass


class TestServerError(NonRetriableError):
    pass


def call_with_retries(func: Callable, n_retries: int = 10, initial_delay: float = 1.0) -> Any:
    """Call the function with retries.

    Args:
        func (Callable): function to call
        n_retries (int, optional): number of retries attempts. Defaults to 10.
        initial_delay (float, optional): delay in sec between attempts. Defaults to 1.0.

    Returns:
        Any: output of the function
    """
    i = 0
    while True:
        try:
            i += 1
            return func()
        except NonRetriableError:
            raise
        except ServerUnavailableError as e:
            error_msg = str(e)
            if "Our servers are currently at full capacity" in error_msg:
                match = re.search(r"Retry in (\d+) s", error_msg)
                if match:
                    delay = int(match.group(1))
                    logger.warning(
                        f"Failed to call function with exception: Our servers are currently at full capacity. Retrying in {delay} sec"
                    )
                    time.sleep(delay)
                else:
                    raise
            else:
                if i >= n_retries:
                    raise

                delay = initial_delay * (2**i)
                logger.warning(f"Failed to call function with exception: {e}. Retrying in {delay:.1f} sec")
                time.sleep(delay)
        except Exception as e:
            if i >= n_retries:
                raise

            delay = initial_delay * (2**i)
            logger.warning(f"Failed to call function with exception: {e}. Retrying in {delay:.1f} sec")
            time.sleep(delay)


def robust_internet_speed() -> tuple[float | None, float | None, float | None]:
    try:
        return call_with_retries(test_internet_speed)
    except Exception:
        logger.error("An error occurred during the speed test, skipping")
        return (None, None, None)


def test_internet_speed() -> tuple[float, float, float]:
    """Measure download/upload internet speed."""
    logger.info("Testing internet speed...")

    st = speedtest.Speedtest(secure=True)
    try:
        st.get_best_server(SPEEDTEST_SERVERS)
        logger.info(f"Best speed test server: {st.best['country']}")
    except Exception:
        pass

    # Perform the download speed test
    download_speed = st.download() / 1000000  # Convert to Mbps

    # Perform the upload speed test
    upload_speed = st.upload() / 1000000  # Convert to Mbps

    # Latency
    latency = float(st.results.ping)  # ms

    # Print the results
    logger.info(f"Download Speed: {download_speed:.2f} Mbps")
    logger.info(f"Upload Speed: {upload_speed:.2f} Mbps")
    logger.info(f"Latency: {latency:.2f} ms")

    return (download_speed, upload_speed, latency)


def get_device_info() -> tuple[str, float, float]:
    """Get device name and memory."""
    ram = float(psutil.virtual_memory().total) / 1024**3

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_info = torch.cuda.get_device_properties(device_id)
        device_name = device_info.name
        gpu_memory = device_info.total_memory / 1024**3  # GB
        return device_name, gpu_memory, ram

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        # MPS uses unified memory on Apple Silicon, so RAM is the relevant capacity signal.
        logger.info("CUDA is not available. Using Apple Metal (MPS).")
        return "mps", ram, ram

    logger.error("Neither CUDA nor Apple Metal (MPS) is available. Exiting run.")
    exit(1)


def get_node_info() -> NodeInfo:
    """Collect information about the node."""
    device_name, gpu_memory, ram = get_device_info()
    download_speed, upload_speed, latency = robust_internet_speed()

    node_info = NodeInfo(
        device_name=device_name,
        gpu_memory=gpu_memory,
        ram=ram,
        download_speed=download_speed,
        upload_speed=upload_speed,
        latency=latency,
    )
    return node_info
