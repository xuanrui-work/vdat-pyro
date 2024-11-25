import subprocess
import sys
import argparse

def run_command_repeatedly(command, n):
    for i in range(1, n + 1):
        print(f"\nRunning command iteration {i}/{n}: {' '.join(command)}\n{'='*40}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
            process.wait()

            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                for line in iter(process.stderr.readline, ''):
                    sys.stderr.write(line)
                    sys.stderr.flush()
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        else:
            print(f"\nCompleted iteration {i}/{n}\n{'='*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N',
        type=int,
        help='The number of times to repeat command.'
    )
    parser.add_argument(
        'command',
        type=str,
        help='The command to repeatly run for.'
    )
    args = parser.parse_args()

    run_command_repeatedly(args.command, args.N)
