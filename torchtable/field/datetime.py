from .core import *

class DatetimeFeatureField(Field):
    """
    A generic field for constructing features from datetime columns.
    Args:
        func: Feature construction function
    """
    def __init__(self, func: Callable[[pd.Series], pd.Series], fill_missing: Optional[str]=None,
                 name=None, is_target=False, continuous=False, metadata: dict={}):
        pipeline = (LambdaOperator(lambda s: pd.to_datetime(s))
                    > FillMissing(method=fill_missing)
                    > LambdaOperator(lambda s: func(s.dt)))
        super().__init__(pipeline, name=name, is_target=is_target, continuous=continuous,
                         categorical=not continuous, metadata=metadata)

class DayofWeekField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.dayofweek, **kwargs)

class DayField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.day, **kwargs)

class MonthStartField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.is_month_start, continuous=False, **kwargs)

class MonthEndField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.is_month_end, **kwargs)

class HourField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.hour, **kwargs)

def date_fields(**kwargs) -> List[DatetimeFeatureField]:
    """The default set of fields for feature engineering using a field with date information"""
    return [DayofWeekField(**kwargs), DayField(**kwargs),
            MonthStartField(**kwargs), MonthEndField(**kwargs),
           ]

def datetime_fields(**kwargs) -> List[DatetimeFeatureField]:
    """The default set of fields for feature engineering using a field with date and time information"""
    return [DayofWeekField(**kwargs), DayField(**kwargs),
            MonthStartField(**kwargs), MonthEndField(**kwargs),
            HourField(**kwargs),
           ]


