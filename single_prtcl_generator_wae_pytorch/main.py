from single_prtcl_generator_wae_pytorch.trainer import Trainer

### TRAINING
trainer = Trainer()
autoenc, embedder, deembedder = trainer.train(epochs=20, load=False)
#autoenc = trainer.create_autoenc()
#embedder = trainer.create_embedder()
#deembedder = trainer.create_train_deembedder(embedder=embedder, epochs=500)

#trainer.show_real_gen_data_comparison(autoenc, embedder, load_model=True, save=True)
