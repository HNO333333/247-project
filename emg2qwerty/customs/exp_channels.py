import subprocess
import logging
import os
import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(
    os.path.join(
        "./logs/exp-logs",
        f"exp_channels_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    )
)
logger.addHandler(file_handler)


def main():

    overrides_list = [
        {
            "random_channel_drop.drop_num_channels": i,
            "module.in_features": 33 * (16 - i),
            "exp_name": f"drop_channel",
            "module.num_channels": 16 - i,
        }
        for i in range(1, 9)
    ]

    # Define a function to get checkpoint paths based on test_name
    for overrides in overrides_list:
        # Create the command with overrides
        override_args = " ".join([f"{key}={value}" for key, value in overrides.items()])
        command = f"/home/shuhan/miniconda3/envs/emg/bin/python -m emg2qwerty.customs.train_conformer -cn conformer_drop_channel {override_args}"

        logger.info("===" * 10)
        logger.info(
            f"drop {overrides['random_channel_drop.drop_num_channels']} channels"
        )

        result = subprocess.run(command, shell=True, capture_output=False, text=True)


if __name__ == "__main__":
    main()
