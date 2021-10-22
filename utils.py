import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import requests
import io
import ipywidgets as widgets
output = widgets.Output()
from IPython.display import display, clear_output, Javascript
from random import uniform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.metrics import mean_squared_error

class ContentManager(object):
    def __init__(self,this_window_width = 800):
        self._this_window_width = this_window_width
        self.x_column_list = [ "X%d"%i for i in range(6, 11) ]
        self.x_column_name_maplist = {
            "X6":"溫度1床材", 
            "X7":"溫度2爐下", 
            "X8":"溫度3爐中下",
            "X9":"溫度4爐中上", 
            "X10":"溫度5爐上",
        }

        self.y_column_maplist = {
            "氧 O2": 'Y1',
            "二氧化碳 CO2": 'Y2',
            "一氧化碳 CO_6% O2": 'Y3',
            "氮氧化合物 NOx_6% O2": 'Y4',
            "二氧化硫 SO2": "Y5"
        }
        self.y_column_name_maplist = {
            'Y1': "氧 O2" ,
            'Y2': "二氧化碳 CO2" ,
            'Y3': "一氧化碳 CO_6% O2" ,
            'Y4': "氮氧化合物 NOx_6% O2" ,
            "Y5": "二氧化硫 SO2"
        }
    
        
        ## 原始資料
        self.origin_data = self._getOriginData() 
        
        ## 測試期間
        self.test_period = 30
        
        ## 移動窗格大小
        self.moving_window_size = 360
        
        ## y 欄位
        self.y_column = 'Y3'
        
        ## 決策樹深度參數
        self.tree_max_depth = 3
        
        ## 參數初始化
        self.train_data = None
        self.test_data = None
        self.train_y = None
        self.train_x = None
        self.test_y = None
        self.test_x = None
        self._buildModelDataset()
        
        self._model = None
        
        ## 驗證資料        
        self._valid_data_maplist = {}
        self.val_y = None
        self.val_x = None

    def _getOriginData(self):
        try:
            ### For colab
            return pd.read_csv('/content/fbc_demo/fbc_demo.csv')
        except FileNotFoundError:
            ### For local 端
            return pd.read_csv('fbc_demo.csv')
        except:
            print('讀取網路資源')
            return pd.read_csv(
                io.StringIO(
                    requests.get('https://recognise.trendlink.io/model/fbc_demo.csv', verify=False).content.decode('utf-8')
                ))
    def _trainModel(self):
        self._model = DecisionTreeRegressor(max_depth=self.tree_max_depth)
        self._model.fit(
            self.train_x.values, 
            self.train_y.values
        )
            
    def _buildModelDataset(self):
        data = self.origin_data.copy()

        ##模型輸入特徵lag處理
        self.feature_col = []
        for lag in range(self.moving_window_size+1):
            ## y 
            if lag>0:
                _y_lag = "%s_%d"%(self.y_column, lag)
                data[_y_lag] = data[self.y_column].shift(lag)
                self.feature_col.append(_y_lag)
            
            for x_column in self.x_column_list:
                _x_lag = "%s_%d"%(x_column, lag)
                data[_x_lag] = data[x_column].shift(lag)
                self.feature_col.append(_x_lag)
                
        data = data.dropna()
        self.train_data = data.iloc[:-self.test_period,:]
        self.test_data = data.iloc[-self.test_period:,:]
        self.train_y = self.train_data[[self.y_column]]
        self.train_x = self.train_data[self.feature_col]
        self.test_y  = self.test_data[[self.y_column]]
        self.test_x  = self.test_data[self.feature_col]
        #print('Training index : ',  train_data.index[0], "~", train_data.index[-1])
        #print('Testing  index : ',  test_data.index[0], "~", test_data.index[-1])
        
    def plotXDataDashboard(self):
        display_data = self.test_data.copy()
        display_data['time'] = range(1, display_data.shape[0]+1)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]]
        )

        # X6 ~ X10
        for x_col in self.x_column_list:
            fig.add_trace(
                go.Scatter(x=display_data['time'], y=display_data[x_col], name=x_col, mode='lines+markers'),
                row=2, col=1
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        'time',
                        "X6<br>溫度1床材", 
                        "X7<br>溫度2爐下", 
                        "X8<br>溫度3爐中下",
                        "X9<br>溫度4爐中上", 
                        "X10<br>溫度5爐上"
                    ],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[display_data['time'].tolist()]+[display_data[x_col].tolist() for x_col in self.x_column_list],
                    align = "left")
            ),
            row=1, col=1
        )

        fig.update_layout(
            width=self._this_window_width,
            showlegend=True,
            title_text="燃燒爐資料-輸入",
            legend=dict(y=0.5, traceorder='reversed', font_size=16)
        )

        fig.show()
        
    def plotYDataDashboard(self):
        display_data = self.test_data.copy()
        display_data['time'] = range(1, display_data.shape[0]+1)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]]
        )

        # Y1 ~ Y5
        for y_col in list(self.y_column_name_maplist.keys()):
            fig.add_trace(
                go.Scatter(
                    x=display_data['time'], 
                    y=display_data[y_col], 
                    name=y_col, 
                    mode='lines+markers',
                    visible= 'legendonly' if y_col != 'Y3' else None
                ),
                row=2, col=1
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['time']+[
                        "%s<br>%s"%(y_col, self.y_column_name_maplist[y_col]) for y_col in list(self.y_column_name_maplist.keys())
                    ],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[display_data['time'].tolist()]+[display_data[y_col].tolist() for y_col in list(self.y_column_name_maplist.keys())],
                    align = "left")
            ),
            row=1, col=1
        )

        fig.update_layout(
            width=self._this_window_width,
            showlegend=True,
            title_text="燃燒爐資料-輸出",
            legend=dict(y=0.5, traceorder='reversed', font_size=16)
        )

        fig.show()

    def plotPredictionData(self): 
        display_data = self.test_data.copy()
        display_data['time'] = range(1, display_data.shape[0]+1)
                
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]]
        )
        
        _table_header = ['time', self.y_column]
        _table_value =[
            display_data['time'].tolist(),
            display_data[self.y_column].tolist(),
        ]
        msr = None
        if not self._model is None:
            test_prediction = self._model.predict(self.test_x.values)
            display_data['pred'] = test_prediction
            msr = mean_squared_error(display_data[self.y_column], test_prediction)
            # pred Y
            fig.add_trace(
                go.Scatter(
                    x=display_data['time'], 
                    y=display_data['pred'], 
                    name= f'預測的{self.y_column_name_maplist[self.y_column]}', 
                    mode='lines+markers',
                    visible= 'legendonly',
                    marker_color='rgba(240, 52, 52, 1)'
                ),
                row=2, col=1
            )
            _table_header.append('pred')
            _table_value.append(display_data['pred'].tolist())
            
        
        # real Y
        fig.add_trace(
            go.Scatter(
                x=display_data['time'], 
                y=display_data[self.y_column], 
                name=f'真實的{self.y_column_name_maplist[self.y_column]}', 
                mode='lines+markers',
                marker_color='rgba(44, 130, 201, 1)'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=_table_header,
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=_table_value,
                    align = "left")
            ),
            row=1, col=1
        )

        if not msr is None:
            title_text = "%s-預測 , 錯誤率：(%.3f)"%(self.y_column_name_maplist[self.y_column], msr)
        else:
            title_text = "%s-預測 "%(self.y_column_name_maplist[self.y_column])
            
        fig.update_layout(
            width=self._this_window_width,
            showlegend=True,
            title_text=title_text,
            legend=dict(y=0.5, traceorder='reversed', font_size=16)
        )

        fig.show()

    def plotFeatureImportances(self, top_k=10):
        feature_importances = pd.DataFrame(
            self._model.feature_importances_, 
            index=self.train_x.columns, 
            columns=['value']
        ).sort_values('value', ascending=False)
        feature_importances['name'] = feature_importances.index
        feature_importances = feature_importances.query("value > 0")
        if feature_importances.shape[0] < top_k:
            top_k =  feature_importances.shape[0]
        fig = px.pie(feature_importances.head(top_k), values='value',  names='name', title=f'前{top_k}個重要的特徵')
        fig.show()
    
    def plotTree(self):
        if self.tree_max_depth<5:
            plt.figure(figsize=(40,20))
            _ = tree.plot_tree(
                self._model, 
                feature_names=self.train_x.columns,
                filled=True
            )
        
    ### valid data
    def makeValidData(self, col_name, start_v, end_v):
        self._valid_data_maplist[col_name] = np.array([
            uniform(start_v, end_v) for _ in range(
                self.moving_window_size+ self.test_period
            )
        ])
        
    def buildValidDataset(self):
        data = pd.DataFrame(self._valid_data_maplist)
        
        ##模型輸入特徵lag處理
        for lag in range(self.moving_window_size+1):
            ## y 
            if lag>0:
                _y_lag = "%s_%d"%(self.y_column, lag)
                data[_y_lag] = data[self.y_column].shift(lag)
            
            for x_column in self.x_column_list:
                _x_lag = "%s_%d"%(x_column, lag)
                data[_x_lag] = data[x_column].shift(lag)
        

        data = data.dropna()
        self.val_data = data
        self.val_y  = self.val_data[[self.y_column]]
        self.val_x  = self.val_data[self.feature_col]
         
    def plotVaildData(self):
        display_data = self.val_data.copy()
        display_data['time'] = range(1, display_data.shape[0]+1)
        test_prediction = self._model.predict(self.val_x.values)
        display_data['pred'] = test_prediction
        
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]]
        )
        
        ## valid table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        'time',
                        "X6<br>溫度1床材", 
                        "X7<br>溫度2爐下", 
                        "X8<br>溫度3爐中下",
                        "X9<br>溫度4爐中上", 
                        "X10<br>溫度5爐上",
                        f'真實的{self.y_column}',
                        f'預測的{self.y_column}'
                    ],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[display_data['time'].tolist()]+\
                    [display_data[x_col].tolist() for x_col in self.x_column_list]+\
                    [display_data[self.y_column].tolist(), 
                    display_data['pred'].tolist()
                    ],
                    align = "left")
            ),
            row=1, col=1
        )

        # X6 ~ X10
        for x_col in self.x_column_list:
            fig.add_trace(
                go.Scatter(x=display_data['time'], y=display_data[x_col], name=x_col, mode='lines+markers'),
                row=2, col=1
        )

        # real Y
        fig.add_trace(
            go.Scatter(
                x=display_data['time'], 
                y=display_data[self.y_column], 
                name=f'真實的{self.y_column}', 
                mode='lines+markers',
                marker_color='rgba(44, 130, 201, 1)'
            ),
            row=3, col=1
        )
        # pred Y
        fig.add_trace(
            go.Scatter(
                x=display_data['time'], 
                y=display_data['pred'], 
                name= f'預測的{self.y_column}', 
                mode='lines+markers',
                visible= 'legendonly',
                marker_color='rgba(240, 52, 52, 1)'
            ),
            row=3, col=1
        )        

        fig.update_layout(
            width=self._this_window_width,
            showlegend=True,
            title_text="模擬燃燒爐資料",
            legend=dict(y=0.5, traceorder='reversed', font_size=16)
        )

        fig.show()

    ## UI 事件
    def testPeriodOnChange(self, change):
        if change['name'] == 'value':
            self.test_period = change['new']
            
    def movingWindowSizeOnChange(self, change):
        if change['name'] == 'value':
            self.moving_window_size = change['new']
            self._model = None
            
    def treeMaxDepthOnChange(self, change):
        if change['name'] == 'value':
            self.tree_max_depth = change['new']
            self._model = None

    def yColumnOnChange(self, change):
        if change['name'] == 'value':
            self.y_column = self.y_column_maplist[change['new']]
            self._model = None
                  
    def modelPredictionButtonOnClick(self, button_event):
        self._buildModelDataset()
        self._trainModel()
        
            
class DisplayManager(object):
    def __init__(self, output, content_manager: ContentManager):
        self._output = output
        self._content_manager = content_manager
        
        self._test_period_widget = widgets.IntText(
            value=content_manager.test_period,
            disabled=False
        )
        self._moving_window_size_widget = widgets.IntText(
            value=content_manager.moving_window_size,
            disabled=False
        )
        self._y_column_widget = widgets.RadioButtons(
            options=list(content_manager.y_column_maplist.keys()),
            value=content_manager.y_column_name_maplist[content_manager.y_column],
            layout={'width': 'max-content'}
        )

        self._real_prediction_button = widgets.Button(description="訓練模型")
        self._valid_prediction_button = widgets.Button(description="模擬資料預測")
        
        self._tree_max_depth_widget = widgets.IntText(
            value=content_manager.tree_max_depth,
            disabled=False
        )
        
        ## observe event
        self._test_period_widget.observe(content_manager.testPeriodOnChange)
        self._moving_window_size_widget.observe(content_manager.movingWindowSizeOnChange)
        self._y_column_widget.observe(self._yColumnOnChange)
        self._tree_max_depth_widget.observe(content_manager.treeMaxDepthOnChange)        
        self._real_prediction_button.on_click(self._realPredictionButtonOnClick)
        self._valid_prediction_button.on_click(self._validPredictionButtonOnClick)
        
        
        ## valid data
        self._y_fr = self._makeFloatRangeSlider(self._content_manager.y_column)
        self._y_fr.observe(self._handleSliderChange)
        
        valid_column_list = self._content_manager.x_column_list
        self._valid_fr_list = []
        for v_col in valid_column_list:
            fr = self._makeFloatRangeSlider(v_col)
            fr.observe(self._handleSliderChange)
            self._valid_fr_list.append(fr)
            
    def _makeFloatRangeSlider(self, col_name):
        col_name_value = self._content_manager.train_data[col_name]
        
        q_25 = col_name_value.quantile(0.25)
        q_75 = col_name_value.quantile(0.75)
        max_v = col_name_value.max()
        min_v = col_name_value.min()
        
        float_range_slider = widgets.FloatRangeSlider(
            value=[q_25, q_75],
            min= min_v,
            max= max_v,
            step=0.1,
            description=col_name,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f'
        )
        
        self._content_manager.makeValidData(col_name, q_25, q_75)
        return float_range_slider
    
    def _handleSliderChange(self, change):
        if change['name'] == '_property_lock' and len(change['new'].keys()) == 0:
            owner = change['owner']
            #self._content_manager.makeValidData(
            #    owner.description,
            #    owner.value[0],
            #    owner.value[1]
            #)

    def _testPeriodOnChange(self, change):
        if change['name'] == 'value':
            self._content_manager.content_manager(change)
        
    def _realPredictionButtonOnClick(self, change):
        self._content_manager.modelPredictionButtonOnClick(change)
        clear_output()
        self.displayHyperParamDashboard()
        self._content_manager.plotPredictionData()
        self._content_manager.plotTree()
        self._content_manager.plotFeatureImportances()
        
    
    def _yColumnOnChange(self, change):
        if change['name'] == 'value':
            self._content_manager.yColumnOnChange(change)
            self._y_fr = self._makeFloatRangeSlider(self._content_manager.y_column)
            self._y_fr.observe(self._handleSliderChange)
        
        
    def displayHyperParamDashboard(self):
        display(widgets.Box([
            widgets.Label(value='想預測幾筆：'),
                self._test_period_widget
        ]), self._output)
        display(widgets.Box([
            widgets.Label(value='要考慮多少資料：'),
                self._moving_window_size_widget
        ]), self._output)
        display(widgets.Box(
            [
                widgets.Label(value='想預測的欄位：'),
                self._y_column_widget
            ]
        ), self._output)
        display(widgets.Box(
            [
                widgets.Label(value='模型參數：'),
                self._tree_max_depth_widget
            ]
        ), self._output)
        display(self._real_prediction_button, self._output)
        
    def displayValidDashboard(self):
        for fr in self._valid_fr_list:
            display(fr, self._output)
        display(self._y_fr, self._output)
        display(self._valid_prediction_button, self._output)
            
    def _validPredictionButtonOnClick(self, b):
        if self._content_manager._model is None:
            clear_output()
            self.displayValidDashboard()
            print("請先訓練模型")
        else:
            ### 重build valid data
            self._content_manager._valid_data_maplist = {}
            for fr in self._valid_fr_list:
                self._content_manager.makeValidData(
                    fr.description,
                    fr.value[0],
                    fr.value[1]
                )
            self._content_manager.makeValidData(
                self._y_fr.description,
                self._y_fr.value[0],
                self._y_fr.value[1]
            )
            self._content_manager.buildValidDataset()
            clear_output()
            self.displayValidDashboard()
            self._content_manager.plotVaildData()

            
def getWindowWidth():
    Javascript("""
    var w = window.screen.width;
    IPython.notebook.kernel.execute("this_window_width="+w);
    """)
    if not 'this_window_width' in dir():
        this_window_width = 1920
    return this_window_width
        
    
    
