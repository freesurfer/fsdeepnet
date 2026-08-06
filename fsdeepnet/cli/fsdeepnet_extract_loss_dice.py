import sys
import re

"""
The script extracts training/validation loss and average dice from given fsdeepnet_train.log for each epoch.
The extracted training loss and average dice are sent to stdout.

Example of log entry:
    '2025-09-16 07:26:47,431 [INFO] Epoch [ 39/300], Train Loss: 0.1552, Train Dice Avg: 0.8264'
Pattern to search:
    'Train Loss: (\d+.\d{4}), Train Dice Avg: (\d+.\d{4})'

Usage:
    extract_train_loss_dice.py <path_to_training_log> <train | val>
"""

logfile = sys.argv[1]
phase   = 'Train'
if (len(sys.argv) > 2):
    phase = sys.argv[2]
assert (phase in ["Train", "Val"]), f"Unknown phase '{phase}' - 'Train' and 'Val' are supported"

# regex object
if (phase == 'Train'):
    lossDiceavgRegex = re.compile(r'Train Loss: (\d+.\d{4}), Train Dice Avg: (\d+.\d{4})')
else:
    lossDiceavgRegex = re.compile(r'Val Loss: (\d+.\d{4}), Val Dice Avg: (\d+.\d{4})')
    
with open(logfile, 'r') as file:
    print(f'{phase}_loss', f'{phase}_dice')

    # Process each line here    
    for line in file:
        # .strip() removes leading/trailing whitespace, including newline characters
        line.strip()
        mo = lossDiceavgRegex.search(line)
        if (mo is not None):
            loss = mo.groups()[0]
            dice_avg = mo.groups()[1]
            print(loss, dice_avg)


