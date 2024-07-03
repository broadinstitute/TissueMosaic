# integration tests
from tissuemosaic.model_utils.dino import DinoBenchmarkModel
from tissuemosaic.model_utils.vae import VaeBenchmarkModel
from tissuemosaic.main_original import parse_args, run_simulation
from tissuemosaic.data_utils.datamodule import DummyDM


def test_dino(dummy_dino_dm, trainer):
    datamodule_ch_in = dummy_dino_dm.ch_in

    config_dict = DinoBenchmarkModel.get_default_params()
    config_dict["image_in_ch"] = datamodule_ch_in

    dino_model = DinoBenchmarkModel(**config_dict)
    assert isinstance(dino_model, DinoBenchmarkModel)
    # TODO: Use mock. The test stalls
    # trainer.fit(model=dino_model, datamodule=dummy_dino_dm)


def test_vae(dummy_dino_dm, trainer):
    datamodule_ch_in = dummy_dino_dm.ch_in

    config_dict = VaeBenchmarkModel.get_default_params()
    config_dict["image_in_ch"] = datamodule_ch_in

    vae_model = VaeBenchmarkModel(**config_dict)
    assert isinstance(vae_model, VaeBenchmarkModel)

    # TODO: Use mock. The test stalls
    # trainer.fit(model=vae_model, datamodule=dummy_dino_dm)


def test_parse_args(dummy_dino_dm):
    # set few parameters manually and let the other be their default values
    config_dict = parse_args(['--dataset', 'dummy_dm', '--max_epochs', '2', '--check_val_every_n_epoch', '1'])
    assert config_dict["dataset"] == 'dummy_dm'
    assert config_dict["max_epochs"] == 2
    assert config_dict["check_val_every_n_epoch"] == 1

    # pytest stalls on the next line. Why?
    # run_simulation(config_dict=config_dict, datamodule=dummy_dino_dm)
