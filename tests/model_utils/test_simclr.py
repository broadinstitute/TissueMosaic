from tissuemosaic.model_utils.simclr import SimclrModel


def test_simclr_construction(dummy_dino_dm, trainer_no_logger, capsys):
    # construction DIno with params compatible with dummy_dm
    config = SimclrModel.get_default_params()
    config["image_size"] = dummy_dino_dm.global_size
    config["image_in_ch"] = dummy_dino_dm.ch_in
    simclr = SimclrModel(**config)
    assert isinstance(simclr, SimclrModel)

    # with capsys.disabled():
    #     print(vae)

    trainer_no_logger.fit(model=simclr, datamodule=dummy_dino_dm)