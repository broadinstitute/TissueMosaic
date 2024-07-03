from tissuemosaic.model_utils.dino import DinoModel


def test_dino_construction(dummy_dino_dm, trainer_no_logger, capsys):
    # construction DIno with params compatible with dummy_dm
    config = DinoModel.get_default_params()
    config["image_size"] = dummy_dino_dm.global_size
    config["image_in_ch"] = dummy_dino_dm.ch_in
    dino = DinoModel(**config)
    assert isinstance(dino, DinoModel)

    # with capsys.disabled():
    #     print(vae)

    trainer_no_logger.fit(model=dino, datamodule=dummy_dino_dm)