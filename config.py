import os


class Config:
    # Directories
    data_dir = 'data'

    # Data
    image_size = 384
    input_dim = 3
    num_of_classes = 100

    # Trainer
    train_batch_size = 96
    val_batch_size = 96
    num_workers = 8
    epochs = 40
    lr = 1e-4
    seed = 42
    use_randaug = True
    use_mixup = True
    use_stochastic_depth = True
    mixup_args = {
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': num_of_classes
    }


    # Model
    model_width = 'S'
    model_depth = 60    
    model_name = model_width + str(model_depth)

    if model_width == 'S':
        patch_dim = 384
    elif model_width == 'B':
        patch_dim = 768
    elif model_width == 'L':
        patch_dim = 1024

    conv_stem_hidden_dims = [32, 64, 128]
    conv_stem_layers = 4
    assert len(conv_stem_hidden_dims) + 1 == conv_stem_layers

    column_hidden_dim = patch_dim//3



