import warnings
class FeatInteraction():
    def __init__(self, group, 
                 value = None
                 ,metric = ['sum','min', 'max', 'mean', 'median', 'std'],
                 date_index = None,
                 date_feature = ['15d', '45d' , '1m', '2m', '3m', '6m', '12m', '2y']) -> None:
        if not isinstance(group, list):
            if isinstance(group, str):
                group = [group]
            else:
                group = list(group)
        
        super().__init__()

        self.group = group
        self.value = value
        self.metric = metric
        self.date_index = date_index
        self.date_feature = date_feature
    
    def fit(self, df):
        if not self.value:
            warnings.warn('value are set to None, the metric paramaters will be ignore. The metric will be set to count')

        #df = super().fit(df)
        if self.date_index:
            warnings.warn('The feature creation based using date are not available yet. please staytune for the update')
            self.date_index = None



        if not self.date_index :
            if not self.value:
                grp = df.groupby(self.group).size().reset_index()
                grp.columns = grp + ['count_' + '_'.join(grp)]
                
            else:
                grp = df.groupby(self.group)[self.value].agg(self.metric).reset_index()
                cols = grp.columns[grp.columns.isin(self.group) == False]

                cols = [f'{self.value}_{x}_by_' + '_'.join(self.group) for x in cols]
                grp.columns = self.group + cols
        self.grp = grp
        return self
    

    def transform(self, df):

        res = df.merge(self.grp, how = 'left', on = self.group)

        return res
