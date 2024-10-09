#文件主要用于配置项目路径和预训练语言模型（PLM）的参数。
import os

#定义class类
class Config:
    #据当前工作目录是否包含 adaptive_bitrate_streaming 字符串，
    #决定基础目录 _base_dir 的值。如果包含，则 _base_dir 为空字符串，否则设置为 'adaptive_bitrate_streaming/'。
    #这样做的目的是为了在不同的运行环境下，确保文件路径能正确地解析
    _base_dir = '' if 'adaptive_bitrate_streaming' in os.getcwd() else 'adaptive_bitrate_streaming/'
    #定义了一些基准模型的路径，这些路径是根据 _base_dir 组合而成的。
    baseline_model_paths = {
        'genet': _base_dir + 'data/all_models/genet/nn_model_ep_9900.ckpt',
        'udr_1': _base_dir + 'data/all_models/udr_1/nn_model_ep_57600.ckpt',
        'udr_2': _base_dir + 'data/all_models/udr_2/nn_model_ep_52400.ckpt',
        'udr_3': _base_dir + 'data/all_models/udr_3/nn_model_ep_58000.ckpt',
        'udr_real': _base_dir + 'data/all_models/udr_real/nn_model_ep_49000.ckpt',
    }
    #定义了训练、验证和测试数据的追踪目录。
    #追踪目录指的是存储用于训练、验证和测试的网络追踪数据（trace data）的文件夹路径。
    # 在这段代码中，追踪目录被用来存放不同阶段（训练、验证、测试）所使用的网络追踪数据。
    # 这些追踪数据通常包括时间序列数据，描述网络行为（如带宽、延迟、丢包率等），用于模拟或测试网络性能。
    trace_dirs = {
        'fcc-train': _base_dir + 'data/traces/train/fcc-train/',
        'fcc-valid': _base_dir + 'data/traces/valid/fcc-valid/',
        'fcc-test': _base_dir + 'data/traces/test/fcc-test/',
    }
    #定义了视频大小数据的目录。
    video_size_dirs = {
        'video1': _base_dir + 'data/videos/video1_sizes/',
        'video2': _base_dir + 'data/videos/video2_sizes/',
    }
    #工件和结果目录: 定义了存储实验工件和结果的目录。
    artifacts_dir = _base_dir + 'artifacts/'
    results_dir = artifacts_dir + 'results/'
    exp_pools_dir = artifacts_dir + 'exp_pools/'

    # plm special
    #模型类型和尺寸: 定义了支持的模型类型和尺寸。
    plm_types = ['gpt2', 'llama', 'llava', 't5-lm', 'opt', 'mistral']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    #PLM目录:根据 _base_dir 确定预训练模型存放的目录。                                                     
    plm_dir = _base_dir + ('../../downloaded_plms' if 'adaptive_bitrate_streaming' in _base_dir else '../downloaded_plms')
    plm_ft_dir = _base_dir + 'data/ft_plms'
    #嵌入尺寸：定义了每种模型类型在不同尺寸下的嵌入维度。
    plm_embed_sizes = {
        'gpt2': {
            'base': 1024,
            'small': 768,
            'large': 1280,
            'xl': 1600,
        },
        'llama': {
            'base': 4096,
        },
        't5-lm': {
            'base': 768,
            'small': 512,
            'large': 4096,
            'xl': 2048,
        },
        'llava': {
            'base': 4096,
        },
        'mistral': {
            'base': 4096,
        },
        'opt': {
            'large': 5120,
            'base': 4096,
            'small': 2560,
            'xs': 2048,
            'xxs': 512,
        },
    }
    #层数配置：定义了每种模型类型在不同尺寸下的层数。
    plm_layer_sizes = {
        'gpt2': {
            'base': 24,
            'small': 12,
            'large': 36,
            'xl': 48
        },
        'llama': {
            'base': 32,
        },
        't5-lm': { 
            'base': 12,
            'small': 6,
            'large': 24,
            'xl': 24
        },
        'llava': {
            'base': 32,
        },
        'mistral': {
            'base': 32,
        },
        'opt': {
            'large': 40,
            'base': 32,
            'small': 32,
            'xs': 32,
            'xxs': 16,
        },
    }

#实例化配置
cfg = Config()
