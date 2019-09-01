import torch.nn as nn

def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'simple':
        assert(opt.dataset_mode == 'generated_simple')
        from .simple_gan import SimpleModel
        model = SimpleModel()
    elif opt.model == 'test':
        #we only need the foreground and the trimap , hence using a slightly different data loader
        assert(opt.dataset_mode == 'testData')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
