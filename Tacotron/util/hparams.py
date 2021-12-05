class HyperParams:
    '''Hyper parameters'''
    
    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.


    # data
    text_cleaners = ['korean_cleaners']
    mod = 'korean'
    if mod == 'english':
        data = "../Datasets/LJSpeech-1.1"
        script = 'metadata.csv'
        check_point_dir = './checkpoint_eng'
        data_dir = './data_LJ'
        out_dir = './output_eng'
    elif mod == 'korean':
        data = "../Datasets/KSS"
        script = 'transcript.txt'
        check_point_dir = './checkpoint_kor'
        data_dir = './data_KSS'
        out_dir = './output_kor'
    elif mod == 'japanese':
        data = "../Datasets/JSS"
        script = 'transcript.txt'
        check_point_dir = './checkpoint_jp'
        data_dir = './data_JSS'
        out_dir = './output_jp'
        
    max_duration = 10.0
    

    # signal processing
    sr = 22050 # Sample rate.
    n_fft = 1024 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = 256 # samples.
    win_length = 1024 # samples.
    mel_dim = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20
    reduction = 5

    # model
    n_symbols = 70
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    decoder_dim = 256
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    lr = 0.001 # Initial learning rate.
    
        
    pre_cp = '/pre_cp'
    post_cp = '/post_cp'
    sampledir = 'samples'
    batch_size = 32
    max_iter = 200