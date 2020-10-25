from pl_bolts.datamodules import CityscapesDataModule

dm = CityscapesDataModule("../data/CityScapes")
#model = LitModel()

Trainer().fit(model, dm)