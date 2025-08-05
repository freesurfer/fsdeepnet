import sys
import logging
import datetime
import yaml

class Config:
    @staticmethod
    def load(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config


    @staticmethod
    # can't get yaml.safe_dump() to save in the correct format
    def save(config, cwd=None, cmd=None, saveas=None, indent=0, sort_keys=False, debug=False):
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
        Config.dump(config, fp, indent=indent, sort_keys=sort_keys, debug=debug)

        if (fp != sys.stdout):
            fp.close()


    @staticmethod
    def list2dict(in_list):
        out_dict = {}
        for item in in_list:
            if isinstance(item, dict):
                for key, value in item.items():
                    out_dict[key] = value
            else:
                out_dict[item] = {}
                    

        return out_dict


    @staticmethod
    def get_augmentations(aug_list):
        aug_dict = Config.list2dict(aug_list)
        return aug_dict.keys()


    @staticmethod
    # 'sort_keys' is not implemented yet
    # parent = {} (dict), [] (list), () (tuple)
    def dump(data, fp=sys.stdout, indent=0, sort_keys=False, parent=None, listidx=0, debug=False):
        if (debug):
            fp.write(f"\n[DEBUG] indent={indent}, parent={parent}, listidx={listidx}, data={data}\n")
        if isinstance(data, dict):
            for key, value in data.items():
                # print dict key
                if (parent is not None and isinstance(parent, list)):
                    if (listidx == 0):
                        fp.write("\n")
                    fp.write(" " * indent + "- " + str(key) + ": ")
                else:
                    if (parent is None):
                        fp.write("\n")
                    fp.write(" " * indent + str(key) + ": ")

                # print dict value
                if isinstance(value, (dict, list, tuple)):
                    if isinstance(value, dict):
                        fp.write("\n")
                    Config.dump(value, fp, indent + 4, parent={}, debug=debug)
                else:
                    if (value is not None):
                        fp.write(str(value))
                    fp.write("\n")
        elif isinstance(data, (list, tuple)):
            for index, item in enumerate(data):
                # this is the case for augmentation specifications
                if isinstance(item, (dict, list, tuple)):
                    Config.dump(item, fp, indent, parent=[], listidx=index, debug=debug)  # indent+4
                else:
                    if (index == 0):
                        fp.write("[")
                    fp.write(str(item))
                    if (index < len(data)-1):
                         fp.write(", ")
                    if (index == len(data)-1):
                        fp.write("]\n")
        else:
            fp.write(str(data) + "\n")


    @staticmethod
    def load_dataset_list(dataset_list_file):
        with open(dataset_list_file, "r", encoding='utf-8') as file:
            dataset_dict = yaml.safe_load(file)

        return dataset_dict

    @staticmethod
    def retrieve_dataset_cohorts(dataset_dict, cohorts):
        # 'cohorts' needs to be a list
        dataset = []
        for cohort in (cohorts):
            ds_cohort = dataset_dict.get(cohort)
            if (ds_cohort is not None):
                dataset.extend(ds_cohort)

        return dataset
