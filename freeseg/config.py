import sys
import logging
import datetime
import yaml

class Config:
    @staticmethod
    def load(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config


    @staticmethod
    # can't get yaml.safe_dump() to save in the correct format
    def save(config, cwd=None, cmd=None, saveas=None, indent=0, sort_keys=False):
        fp = sys.stdout
        if (saveas is not None):
            logging.info(f"save config yaml {saveas}")
            fp = open(saveas, 'w')

        fp.write("# " + str(datetime.datetime.now()) + "\n");
        if (cwd is not None):
            fp.write(f"# CWD: {cwd}\n")
        if (cmd is not None):
            fp.write(f"# CMD: {cmd}\n")
        fp.write("#\n")

        # output the config
        Config.dump(config, fp, indent=indent, sort_keys=sort_keys)

        if (fp != sys.stdout):
            fp.close()


    @staticmethod
    # 'sort_keys' is not implemented yet
    def dump(data, fp=sys.stdout, indent=0, sort_keys=False):
        if isinstance(data, dict):
            for key, value in data.items():
                fp.write(" " * indent + str(key) + ": ")
                if isinstance(value, (dict, list, tuple)):
                    if isinstance(value, dict):
                        fp.write("\n")
                    Config.dump(value, fp, indent + 4)
                else:
                    fp.write(str(value) + "\n")
        elif isinstance(data, (list, tuple)):
            fp.write("[ ")
            for item in data:
                if isinstance(item, (dict, list, tuple)):
                    Config.dump(item, fp, indent + 4)
                else:
                    fp.write(str(item) + ", ")
            fp.write("]")
        else:
            fp.write(str(data) + "\n")

        fp.write("\n")

