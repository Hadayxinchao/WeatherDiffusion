from .snow100k import Snow100K
from .raindrop import RainDrop
from .outdoorrain import OutdoorRain
from .ohaze import OHaze
from .custom_haze import CustomHaze
from .allweather import AllWeather

def get_dataset(config):
    dataset_name = config.data.dataset
    
    if dataset_name.startswith('CustomHaze'):
        return CustomHaze(config)
    
    elif dataset_name == 'Snow100K':
        return Snow100K(config)
    elif dataset_name == 'RainDrop':
        return RainDrop(config)
    elif dataset_name == 'OutdoorRain':
        return OutdoorRain(config)
    elif dataset_name == 'OHaze':
        return OHaze(config)
    elif dataset_name == 'AllWeather':
        return AllWeather(config)
    else:
        raise KeyError(f"Dataset không hợp lệ: {dataset_name}")
