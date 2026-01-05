import os
import sys
import torch
import logging
import datetime
import yaml
import numpy as np
import shutil

from freeseg.utils import utility as utils

class Config:
    @staticmethod
    def process(args, logger=None, require_train_outfolder=True, require_dataset_list=True, test_augment=False, assert_dimensions=True):
        if (logger is None):
            logger = logging

        ### load config.yaml
        config = Config.load(args.config)

        ### update config with commandline user options and default config.dataloader parameters
        config = Config.update(config, args)

        ### argument checks
        if ('checkpoint' in args and args.checkpoint is not None):
            if not os.path.isfile(args.checkpoint):
                logger.error('ERROR: file does not exist: %s' % args.checkpoint)
                sys.exit(1)

        output_folder = None
        if (require_train_outfolder):
            output_folder = config["training"].get("train_output_folder", None)
            assert (output_folder is not None), "Use '--train_output_folder <>' or 'train_output_folder' in config.yaml to specify training output directory"
        elif (test_augment):
            output_folder = config["preprocessing"].get("augmentation_dir", None)
            assert (output_folder is not None), "Use '--augmentation_dir <>' to specify augmentation output directory"
        if (require_dataset_list):
            assert (config["dataset"].get("dataset_list_file", None) is not None), "Use '--dataset_list_file <dataset.yaml>' or 'dataset_list_file' in config.yaml to specify the dataset"

        if (assert_dimensions):
            crop_size = config["preprocessing"]["crop_size"]
            nb_levels = config["model"]["nb_levels"]
            ndims = config["model"]["ndims"]
            assert (np.all(np.array(crop_size) % (2**(nb_levels)) == 0)), f"crop_size {crop_size} needs to be divisible by 2^{nb_levels}"
            assert (ndims == len(crop_size)), f"crop_size {crop_size} is not for {ndims}D"

        now = datetime.datetime.now()
        dt_nowstring = str(now).replace(' ', '.').replace(':', '.')

        ### setup and configure root and main logger
        if (output_folder is not None):
            output_folder = os.path.abspath(output_folder)
            if (not os.path.exists(output_folder)):
                os.makedirs(output_folder)
        logfile = None
        if (require_train_outfolder):
            logfile = args.logfile if ('logfile' in args and args.logfile is not None) else os.path.join(output_folder, f"log.{dt_nowstring}")
            utils.config_logger(logfile=logfile)

        ### print the command
        cmd = ' '.join(sys.argv)
        cmdopts = cmd.split("--")
        cwd = os.getcwd()
        logger.info("===================== Current date and time: " + str(now) + " =====================")
        logger.info("CWD: " + cwd)
        logger.info("CMD: " + "\n                                    --".join(cmdopts))
        logger.info("PID: " + str(os.getpid()))
        logger.info("")

        ### save updated config and dataset_list_file
        config_saveas, dataset_list_saveas = None, None
        if (output_folder is not None):
            # copy the user input config.yaml
            shutil.copyfile(args.config, os.path.join(output_folder, f"input_config.{dt_nowstring}.yaml"))
            config_saveas = os.path.join(output_folder, f"config.{dt_nowstring}.yaml")
            # save the config updated with command line args
            Config.save(config, cwd=cwd, cmd=cmd, saveas=config_saveas)
            # cpoy dataset_list.yaml
            dataset_list_saveas = os.path.join(output_folder, f"dataset_list.{dt_nowstring}.yaml")
            shutil.copyfile(config["dataset"]["dataset_list_file"], dataset_list_saveas)

        ### IN THE REST OF THE FUNCTION,
        ### CONFIG WILL BE RE-ARRANGED AND UPDATED TO BE USED IN TRAINING SETUP
        ### UPDATE config.dataloader
        batch_size = 1
        if (config.get("training", None)):
            batch_size = config["training"]["batch_size"]
            config["dataloader"].update({"batch_size": batch_size})

        ### UPDATE config.dataset
        config["dataset"].update({ "ndims": config["model"]["ndims"],
                                   "batch_size": batch_size,
                                  #"crop_size": crop_size,                                   
                                 })
    
        ### set training, preprocessing devices
        if ('cpu' in args and args.cpu):
            os.environ["CUDA_VISIBLE_DEVICES"]=""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_index = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
            gpu_index = None
        # force data preprocessing (augmentation) to run on CPU is pin_memory = True or num_worker > 0
        if (config["dataloader"]["pin_memory"] or config["dataloader"]["num_workers"] > 0):
            preprocessing_device = torch.device("cpu")
        else:
            preprocessing_device = device

        ### UPDATE config
        config.update({"cmd": cmd,
                       "cwd": cwd,
                       "now": now,
                       "device": device,
                       "gpu_index": gpu_index,
                       "preprocessing_device": preprocessing_device,
                       "model_checkpoint": args.checkpoint if ('checkpoint' in args) else None,
                       "ctab": args.ctab if ('ctab' in args) else None,
                       "keep_trainset_in_memory": args.keep_trainset_in_memory if ('keep_trainset_in_memory' in args) else False,
                       "logfile": logfile,
                       "output_folder": output_folder,
                       "train_augmentations": None,
                       "vmp": args.vmp if ('vmp' in args) else False,
                       "debug": args.debug if ('debug' in args) else False,
                       "verbose": args.verbose if ('verbose' in args) else False,
                       "train_cohort": args.train_cohort if ('train_cohort' in args) else None,
                       "validation_cohort": args.validation_cohort if ('validation_cohort' in args) else None,
                       "config_saveas": config_saveas,
                       "dataset_list_saveas": dataset_list_saveas})
                
        return config


    @staticmethod
    def load(config_file):
        assert (os.path.isfile(config_file)), f"file {config_file} doesn't exist"

        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config


    @staticmethod
    # update the input config with commandline user options and default config.dataloader parameters
    # NOTES: need to first check if the argument is available because
    #        update(), process() is shared between freeseg_train.py,
    #        test_dataloader.py, test_segmentationdataset.py that have different argument set
    def update(config, args):
        # backward compatibility - handle config.yaml w/o 'dataloader' section
        if ('dataloader' not in config):
            config['dataloader'] = {}

        # overwrite config with command line options
        if ('verbose' in args and args.verbose):  # bool
            config["preprocessing"]["verbose"] = args.verbose
        if ('model_name' in args and args.model_name is not None):
            config["model"]["name"] = args.model_name
        if ('weight_init' in args and args.weight_init is not None):
            config["model"]["weight_init"] = args.weight_init
        if ('nb_labels' in args and args.nb_labels is not None):
            config["model"]["nb_labels"] = args.nb_labels
        if ('nb_levels' in args and args.nb_levels is not None):
            config["model"]["nb_levels"] = args.nb_levels
        if ('nb_features' in args and args.nb_features is not None):
            config["model"]["nb_features"] = args.nb_features
        if ('feat_mult' in args and args.feat_mult is not None):
            config["model"]["feat_mult"] = args.feat_mult
        if ('nb_conv_per_level' in args and args.nb_conv_per_level is not None):
            config["model"]["nb_conv_per_level"] = args.nb_conv_per_level
        if ('conv_size' in args and args.conv_size is not None):
            config["model"]["conv_size"] = args.conv_size
        if ('pool_size' in args and args.pool_size is not None):
            config["model"]["pool_size"] = args.pool_size
        if ('use_residuals' in args and args.use_residuals): # bool
            config["model"]["use_residuals"] = args.use_residuals
        if ('wl2_epochs' in args and args.wl2_epochs is not None):
            config["training"]["wl2_epochs"] = args.wl2_epochs
        if ('dice_epochs' in args and args.dice_epochs is not None):
            config["training"]["dice_epochs"] = args.dice_epochs
        if ('learning_rate' in args and args.learning_rate is not None):
            config["training"]["learning_rate"] = args.learning_rate
        if ('dataset_list_file' in args and args.dataset_list_file is not None):
            config["dataset"]["dataset_list_file"] = args.dataset_list_file
        if ('crop_size' in args and args.crop_size is not None):
            config["preprocessing"]["crop_size"] = args.crop_size
        if ('train_output_folder' in args and args.train_output_folder is not None):
            config["training"]["train_output_folder"] = args.train_output_folder
        if ('report_moving_avg' in args and args.report_moving_avg): # bool
            config["training"]["report_moving_avg"] = args.report_moving_avg        
        if ('augmentation_dir' in args and args.augmentation_dir is not None):
            config["preprocessing"]["augmentation_dir"] = args.augmentation_dir
        if ('deterministic' in args and args.deterministic): # bool
            config["training"]["deterministic"] = args.deterministic
        if ('batch_size' in args and args.batch_size is not None):
            config["training"]["batch_size"] = args.batch_size
        if ('write_tensorboard_summary' in args and args.write_tensorboard_summary): # bool
            config["training"]["write_tensorboard_summary"] = args.write_tensorboard_summary
        if ('perform_evaluation' in args and args.perform_evaluation): # bool
            config["training"]["perform_evaluation"] = args.perform_evaluation
        if ('best_model_metric' in args and args.best_model_metric is not None):
            config["training"]["best_model_metric"] = args.best_model_metric
        if ('num_workers' in args and args.num_workers is not None):
            config["dataloader"]["num_workers"] = args.num_workers
        if ('prefetch_factor' in args and args.prefetch_factor is not None):
            config["dataloader"]["prefetch_factor"] = args.prefetch_factor
        if ('pin_memory' in args and args.pin_memory): # bool
            config["dataloader"]["pin_memory"] = args.pin_memory
        if ('persistent_workers' in args and args.persistent_workers): # bool
            config["dataloader"]["persistent_workers"] = args.persistent_workers
        if ('res_diff_thresh' in args and args.res_diff_thresh is not None):
            config["dataset"]["res_diff_thresh"] = args.res_diff_thresh

        # update config.dataloader
        num_workers = config["dataloader"].get("num_workers", 0)
        pin_memory = config["dataloader"].get("pin_memory", False)
        persistent_workers = config["dataloader"].get("persistent_workers", False)
        prefetch_factor = config["dataloader"].get("prefetch_factor", 2)
        if (num_workers == 0):
            prefetch_factor = None
            persistent_workers = False
        config["dataloader"].update({"num_workers": num_workers,
                                     "pin_memory": pin_memory,
                                     "persistent_workers": persistent_workers,
                                     "prefetch_factor": prefetch_factor})

        return config


    @staticmethod
    def print(cfg, logger=None):
        if (logger is None):
            logger = logging

        logger.info("")
        logger.info("Training Device: {}".format(cfg['device']) + (f' (GPU index: {cfg["gpu_index"]})' if (cfg.get('gpu_index') is not None) else ''))
        if (cfg["model_checkpoint"] is not None):
            logger.info(f"resume training from model: {cfg['model_checkpoint']}")
        logger.info(f"optimizer: {cfg['training'].get('optimizer', 'torch.optim.Adam')}")
        logger.info(f"data_generator: {cfg['training'].get('data_generator', None)}")
        if (cfg["training"].get("wl2_epochs", 0) > 0):
            logger.info(f"wl2_epochs: {cfg['training'].get('wl2_epochs')}")
            logger.info(f"wl2_metrics: {cfg['training'].get('wl2_metrics', None)}")
            logger.info(f"pre_train_learning_rate: {cfg['training']['pre_train_learning_rate']}")
        if (cfg["training"].get("dice_epochs", 0) > 0):
            logger.info(f"dice_epochs: {cfg['training'].get('dice_epochs')}")
            logger.info(f"model_metrics: {cfg['training'].get('model_metrics', None)}")
            logger.info(f"model_metrics_accuracy: {cfg['training'].get('model_metrics_accuracy', None)}")
            logger.info(f"learning_rate: {cfg['training']['learning_rate']}")
        logger.info(f"steps_per_epoch: {cfg['training']['steps_per_epoch']}")
        logger.info(f"report_moving_avg: {cfg['training'].get('report_moving_avg', False)}")
        logger.info(f"batch_size: {cfg['training']['batch_size']}")

        logger.info(f"keep_trainset_in_memory: {cfg['keep_trainset_in_memory']}")
        logger.info(f"deterministic: {cfg['preprocessing'].get('deterministic', False)}")
        perform_evaluation = cfg['training'].get('perform_evaluation', False)
        logger.info(f"perform_evaluation: {perform_evaluation}")
        if (perform_evaluation):
            logger.info(f"best_model_metric: {cfg['training'].get('best_model_metric')}")
        logger.info("Preprocessing Device: {}".format(cfg['preprocessing_device']) + (f' (GPU index: {cfg["gpu_index"]})' if (cfg.get('gpu_index') is not None) else ''))
        logger.info(f"Preprocessing augmentation_wrapper: {cfg['preprocessing']['augmentation_wrapper']}")
        logger.info(f"Preprocessing augmentations: {cfg['train_augmentations']}")
        logger.info(f"Preprocessing crop_size: {cfg['preprocessing']['crop_size']}")
        logger.info(f"Preprocessing sampling_hp: {cfg['preprocessing'].get('sampling_hp', True)}")
        logger.info(f"Preprocessing num_workers: {cfg['dataloader']['num_workers']}")
        logger.info(f"Preprocessing persistent_workers: {cfg['dataloader']['persistent_workers']}")
        logger.info(f"Preprocessing pin_memory: {cfg['dataloader']['pin_memory']}")
        logger.info(f"Preprocessing prefetch_factor: {cfg['dataloader']['prefetch_factor']}")

        logger.info(f"color table: {cfg['ctab']}")
        logger.info(f"output_folder: {cfg['output_folder']}")
        logger.info(f"training config: saved as {cfg['config_saveas']}")
        logger.info(f"dataset list: saved as {cfg['dataset_list_saveas']}")
        logger.info(f"log file: {cfg['logfile']}")
        logger.info("")

    
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
            cmdopts = cmd.split("--")
            fp.write("# CMD: " + "\n#              --".join(cmdopts) + "\n\n")

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
