from .snow100k import Snow100K
from .raindrop import RainDrop
from .outdoorrain import OutdoorRain
from .ohaze import OHaze
from .custom_haze import CustomHaze
from .allweather import AllWeather

def get_dataset(args, config):
    dataset_name = config.data.dataset
    
    if dataset_name == 'Snow100K':
        return Snow100K(args, config)
    elif dataset_name == 'RainDrop':
        return RainDrop(args, config)
    elif dataset_name == 'OutdoorRain':
        return OutdoorRain(args, config)
    elif dataset_name == 'OHaze':
        return OHaze(args, config)
    elif dataset_name == 'AllWeather':
        return AllWeather(args, config)
    elif dataset_name.startswith('CustomHaze'):
        return CustomHaze(args, config)
    
    else:
        raise KeyError(f"Dataset không hợp lệ: {dataset_name}. Hãy kiểm tra lại configs.")