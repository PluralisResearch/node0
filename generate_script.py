# Copyright 2025 Pluralis Research

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import re
import stat

from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional
from urllib import request


# fmt: off
script_template_docker = Template('''#!/bin/bash

CONTAINER_NAME_VAR="$container_name"

if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME_VAR}$"; then
    echo "Container $CONTAINER_NAME_VAR already exists. Stopping and removing..."

    docker stop $CONTAINER_NAME_VAR >/dev/null 2>&1
    docker rm $CONTAINER_NAME_VAR >/dev/null 2>&1

    echo "Container $CONTAINER_NAME_VAR stopped and removed."
fi

docker run --name $CONTAINER_NAME_VAR -itd --ipc=host --network=host --gpus device=$gpu_id -v $root_folder:$workdir $image_name >/dev/null 2>&1

echo "Container $CONTAINER_NAME_VAR created."

nohup docker exec -w $workdir $CONTAINER_NAME_VAR bash -c "CUDA_VISIBLE_DEVICES=0 python3.11 $python_file_path \
--run_config $run_config_path \
--token $token \
$email_flag \
--auth_server $auth_server \
--custom_module_path $custom_module_path \
$identity_path_flag \
--host_maddrs /ip4/0.0.0.0/tcp/$host_port \
--announce_maddrs /ip4/$public_ip/tcp/$announce_port \
--initial_peers $peer_addrs" \
> run.out 2>&1 &

echo "Server is started. See run.out for the output."
''')


script_template = Template('''#!/bin/bash

CUDA_VISIBLE_DEVICES=$gpu_id python3.11 $python_file_path \
--run_config $run_config_path \
--token $token \
$email_flag \
--auth_server $auth_server \
--custom_module_path $custom_module_path \
$identity_path_flag \
--host_maddrs /ip4/0.0.0.0/tcp/$host_port \
--announce_maddrs /ip4/$public_ip/tcp/$announce_port \
--initial_peers $peer_addrs \
''')


LIBRARY_NAME = [32, 32, 10, 9608, 9608, 9608, 9559, 9617, 9617, 9608, 9608, 9559, 9617, 9608, 9608, 9608, 9608, 9608, 9559, 9617,
                    9608, 9608, 9608, 9608, 9608, 9608, 9559, 9617, 9608, 9608, 9608, 9608, 9608, 9608, 9608, 9559, 32, 32, 9617, 9608,
                    9608, 9608, 9608, 9608, 9559, 9617, 10, 9608, 9608, 9608, 9608, 9559, 9617, 9608, 9608, 9553, 9608, 9608, 9556, 9552,
                    9552, 9608, 9608, 9559, 9608, 9608, 9556, 9552, 9552, 9608, 9608, 9559, 9608, 9608, 9556, 9552, 9552, 9552, 9552, 9565,
                    32, 32, 9608, 9608, 9556, 9552, 9552, 9608, 9608, 9559, 10, 9608, 9608, 9556, 9608, 9608, 9559, 9608, 9608, 9553, 9608,
                    9608, 9553, 9617, 9617, 9608, 9608, 9553, 9608, 9608, 9553, 9617, 9617, 9608, 9608, 9553, 9608, 9608, 9608, 9608, 9608,
                    9559, 9617, 9617, 32, 32, 9608, 9608, 9553, 9617, 9617, 9608, 9608, 9553, 10, 9608, 9608, 9553, 9562, 9608, 9608, 9608,
                    9608, 9553, 9608, 9608, 9553, 9617, 9617, 9608, 9608, 9553, 9608, 9608, 9553, 9617, 9617, 9608, 9608, 9553, 9608, 9608,
                    9556, 9552, 9552, 9565, 9617, 9617, 32, 32, 9608, 9608, 9553, 9617, 9617, 9608, 9608, 9553, 10, 9608, 9608, 9553, 9617,
                    9562, 9608, 9608, 9608, 9553, 9562, 9608, 9608, 9608, 9608, 9608, 9556, 9565, 9608, 9608, 9608, 9608, 9608, 9608, 9556,
                    9565, 9608, 9608, 9608, 9608, 9608, 9608, 9608, 9559, 32, 32, 9562, 9608, 9608, 9608, 9608, 9608, 9556, 9565, 10, 9562,
                    9552, 9565, 9617, 9617, 9562, 9552, 9552, 9565, 9617, 9562, 9552, 9552, 9552, 9552, 9565, 9617, 9562, 9552, 9552, 9552,
                    9552, 9552, 9565, 9617, 9562, 9552, 9552, 9552, 9552, 9552, 9552, 9565, 32, 32, 9617, 9562, 9552, 9552, 9552, 9552, 9565, 9617]

COMPANY_NAME = [119927, 119949, 119958, 119955, 119938, 119949, 119946, 119956, 32, 119929, 119942, 119956, 119942, 119938, 119955, 119940, 119945]

# fmt: on


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type=str, help="Run config file name")
    parser.add_argument("--token", type=str, help="Huggingface token")
    parser.add_argument("--email", type=str, help="Email address")
    parser.add_argument("--host_port", type=int, default=49200, help="Specify the port to expose")
    parser.add_argument("--announce_port", type=int, default=49200, help="Specify mapped port")
    parser.add_argument("--auth_server", type=str, help="Authentication server URL")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use")
    parser.add_argument("--model_name", type=str, default="llama", help="Name of the model")
    parser.add_argument("--use_docker", action="store_true", help="Run the code inside docker container")
    parser.add_argument("--skip_input", action="store_true", help="Skip interactive arguments input")
    parser.add_argument(
        "--initial_peers",
        type=str,
        nargs="*",
        help="Addresses of active peers in the run",
    )
    parser.add_argument("--identity_path", type=str, help="Path to identity file to be used in P2P")

    args = parser.parse_args()
    return args


def validate_auth_server(auth_server: str) -> str:
    auth_server = auth_server.strip()
    regex = r"https?://*"
    match = bool(re.match(regex, auth_server))
    return auth_server if match else ""


def validate_email(email: str) -> Optional[str]:
    email = email.strip()
    if email == "":
        return ""

    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    match = bool(re.match(regex, email))
    return email if match else None


def validate_initial_peers(initial_peers: str) -> list:
    initial_peers_list = initial_peers.strip().split()
    return initial_peers_list


def validate_yn(yn: str) -> str:
    yn = yn.lower().strip()
    if not yn or yn[0] not in ("y", "n"):
        return ""
    return yn[0]


def validate_number(num: str, min_value: int, max_value: int) -> int:
    try:
        int_num = int(num.strip())
        if not (min_value <= int_num <= max_value):
            return -1
        return int_num
    except Exception:
        return -1


def validate_run_config(run_config: str) -> str:
    name = Path(run_config.strip()).stem
    if not name.startswith("llama"):
        return ""
    return name


def input_arguments(args: argparse.Namespace) -> argparse.Namespace:
    print("".join([chr(c) for c in LIBRARY_NAME]))
    print("".join([chr(c) for c in COMPANY_NAME]))
    print("\n")

    while True:
        if args.token is None:
            token = input("Please enter your HuggingFace token: ")
            while not (token := token.strip()):
                token = input("Token cannot be empty. Please enter your HuggingFace token: ")
            args.token = token

        if args.email is None:
            email = input("Please enter your email [optional]: ")
            while (email := validate_email(email)) is None:
                email = input("Wrong email format. Please enter valid email or leave empty: ")
            args.email = email

        if args.host_port is None:
            host_port = input("Please enter host port to use: ")
            while (host_port := validate_number(host_port, min_value=1, max_value=65535)) == -1:
                host_port = input("Invalid host port. Please enter host port to use: ")
            args.host_port = host_port

        if args.announce_port is None:
            announce_port = input("Please enter announce port to use: ")
            while (announce_port := validate_number(announce_port, min_value=1, max_value=65535)) == -1:
                announce_port = input("Invalid announce port. Please enter announce port to use: ")
            args.announce_port = announce_port

        if args.auth_server is None:
            auth_server = input("Please enter authorization server URL: ")
            while not (auth_server := validate_auth_server(auth_server)):
                auth_server = input("Authorization server URL is invalid. Please enter authorization server URL: ")
            args.auth_server = auth_server

        if args.run_config is None:
            run_config = input("Please enter run configuration file name: ")
            while not (run_config := validate_run_config(run_config)):
                run_config = input("File name is invalid. Please enter run configuration file name: ")
            args.run_config = run_config

        if args.initial_peers is None:
            initial_peers = input("Please enter initial peers (separated by space): ")
            initial_peers.split(" ")
            while not (initial_peers := validate_initial_peers(initial_peers)):
                initial_peers = input(
                    "Initial peers list cannot be empty. Please enter initial peers (separated by space): "
                )
            args.initial_peers = initial_peers

        if args.gpu_id is None:
            gpu_id = input("Please enter GPU ID to use: ")
            while (gpu_id := validate_number(gpu_id, min_value=0, max_value=9999)) == -1:
                gpu_id = input("Invalid GPU ID. Please enter GPU ID to use: ")
            args.gpu_id = gpu_id

        if args.identity_path is None:
            identity_path = input("Please enter identity file path [optional]: ")
            identity_path = identity_path.strip()
            args.identity_path = identity_path if identity_path else None

        if args.use_docker is None:
            use_docker = input(
                "Do you want to run the code inside docker container (needs docker installed and image prebuilt)? [Y/n] "
            )
            while not (use_docker := validate_yn(use_docker)):
                use_docker = input(
                    "Invalid input. Do you want to run the code inside docker container (needs docker installed and image prebuilt)? [Y/n] "
                )

            args.use_docker = True if use_docker == "y" else False

        print("\nThe server will run with the following parameters:")
        print(f"\ttoken: {args.token}")
        print(f"\temail: {args.email}")
        print(f"\thost_port: {args.host_port}")
        print(f"\tannounce_port: {args.announce_port}")
        print(f"\tauth_server: {args.auth_server}")
        print(f"\trun_config: {args.run_config}")
        print(f"\tinitial_peers: {args.initial_peers}")
        print(f"\tgpu_id: {args.gpu_id}")
        print(f"\tidentity_path: {args.identity_path}")
        print(f"\tuse_docker: {args.use_docker}")

        need_change = input("\nDo you want to change anything? [Y/n] ")
        if need_change == "":
            need_change = "y"
        while not (need_change := validate_yn(need_change)):
            need_change = input("Invalid input. Do you want to change anything? [Y/n] ")

        if need_change == "y":
            print("\nPlease select the argument you want to change:")
            print("\t1. token")
            print("\t2. email")
            print("\t3. host_port")
            print("\t4. announce_port")
            print("\t5. auth_server")
            print("\t6. run_config")
            print("\t7. initial_peers")
            print("\t8. gpu_id")
            print("\t9. identity_path")
            print("\t10. use_docker")
            print("\t0. Cancel")

            n_change = input("Enter (0-10): ")
            while (n_change := validate_number(n_change, min_value=0, max_value=10)) == -1:
                n_change = input("Invalid input. Enter (0-10): ")

            if n_change == 1:
                args.token = None
            elif n_change == 2:
                args.email = None
            elif n_change == 3:
                args.host_port = None
            elif n_change == 4:
                args.announce_port = None
            elif n_change == 5:
                args.auth_server = None
            elif n_change == 6:
                args.run_config = None
            elif n_change == 7:
                args.initial_peers = None
            elif n_change == 8:
                args.gpu_id = None
            elif n_change == 9:
                args.identity_path = None
            elif n_change == 10:
                args.use_docker = None
            else:
                pass
            print()
        else:
            print()
            return args


def get_public_ip():
    """Get the public IP address using the ipify API."""
    try:
        with request.urlopen("https://api.ipify.org") as response:
            return response.read().decode("utf-8").strip()
    except Exception as e:
        print(f"Error getting public IP: {e}")
        return None


def main(args: argparse.Namespace):
    # Get the root folder
    root_folder = Path(__file__).resolve().parent

    # Parse run.json
    if Path("run.json").exists():
        with open("run.json") as f:
            exp_config = json.load(f)

            if exp_config["run_config"] and exp_config["run_config"] != "none":
                args.run_config = exp_config["run_config"]

            if exp_config["auth_server"] and exp_config["auth_server"] != "none":
                args.auth_server = exp_config["auth_server"]

            if (
                isinstance(exp_config["seeds"], list)
                and len(exp_config["seeds"]) > 0
                and exp_config["seeds"][0] != "none"
            ):
                args.initial_peers = exp_config["seeds"]

    # Input missing args
    if not args.skip_input:
        args = input_arguments(args)
    else:
        # Check that all parameters are provided
        if args.token is None:
            print("--token is missing")
            exit(1)

        if args.auth_server is None:
            print("auth_server is missing: provide it as an argument --auth_server or define it in run.json")
            exit(1)

        if args.run_config is None:
            print("run_config is missing: provide it as an argument --run_config or define it in run.json")
            exit(1)

        if args.initial_peers is None:
            print("initial_peers is missing: provide it as an argument --initial_peers or define it in run.json")
            exit(1)

    email_flag = f"--email {args.email}" if args.email else ""
    identity_path_flag = f"--identity_path {args.identity_path}" if args.identity_path else ""

    # Get public ip address
    public_ip = get_public_ip()
    if public_ip is None:
        print("Can't get public IP address. Check your Internet connection and try again.")
        exit(1)

    if args.use_docker:
        # Derive full paths
        run_config_path = Path("src", "node0", "configs", f"{args.run_config}.yaml").as_posix()
        custom_module_path = Path("src", "node0", "models", args.model_name, "layers.py").as_posix()
        python_file_path = Path("src", "node0", "run_server.py")

        # Generate container name
        container_name = f"node0_{datetime.now().strftime('%y%m%d%H%M%S')}"

        # Create script from template
        run_script = script_template_docker.safe_substitute(
            {
                "root_folder": root_folder.as_posix(),
                "workdir": "/home/node0",
                "image_name": "pluralis_node0",
                "container_name": container_name,
                "python_file_path": python_file_path,
                "gpu_id": args.gpu_id,
                "run_config_path": run_config_path,
                "token": args.token,
                "host_port": args.host_port,
                "announce_port": args.announce_port,
                "auth_server": args.auth_server,
                "custom_module_path": custom_module_path,
                "identity_path_flag": identity_path_flag,
                "peer_addrs": " ".join(args.initial_peers),
                "public_ip": public_ip,
                "email_flag": email_flag,
            }
        )
    else:
        # Derive full paths
        run_config_path = root_folder.joinpath("src", "node0", "configs", f"{args.run_config}.yaml").as_posix()
        custom_module_path = root_folder.joinpath("src", "node0", "models", args.model_name, "layers.py").as_posix()
        python_file_path = root_folder.joinpath("src", "node0", "run_server.py")

        # Create script from template
        run_script = script_template.safe_substitute(
            {
                "python_file_path": python_file_path,
                "gpu_id": args.gpu_id,
                "run_config_path": run_config_path,
                "token": args.token,
                "host_port": args.host_port,
                "announce_port": args.announce_port,
                "auth_server": args.auth_server,
                "custom_module_path": custom_module_path,
                "identity_path_flag": identity_path_flag,
                "peer_addrs": " ".join(args.initial_peers),
                "public_ip": public_ip,
                "email_flag": email_flag,
            }
        )

    # Save script
    run_script_path = root_folder.joinpath("start_server.sh")

    with open(run_script_path, "w") as f:
        f.writelines(run_script)

    # Make executable
    run_script_path.chmod(run_script_path.stat().st_mode | stat.S_IEXEC)

    print("File start_server.sh is generated. Run ./start_server.sh to join the experiment.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
