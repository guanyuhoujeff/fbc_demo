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
import os
from datetime import datetime, date

class ContentManager(object):
    """本資料共計風速5個欄位；風向4個欄位
    逐十分鐘一筆
    統計資料區間 = 2021/08/01 00:00:01 ~ 2021/10/01 00:00:00


    欄位代碼解說：
    ．儀器類型+高度_N分鐘統計數據
    ．WS = Wind Speed（風速計，單位：m/s = 公尺/秒）
    ．WD = Wind Direction（風向計，單位：度）
    ．10mAVG = 10分鐘平均值

    例如：WS95A_10mAVG ，即代表「95公尺高度風速計A的10分鐘統計資料」
    ※僅95公尺風速計有2支，所以最後會多帶有A、B之尾碼 
    """
    def __init__(self,this_window_width = 800):
        self.feature_list = [
            {'name': "10公尺風速器", 'type': 'ws', 'column' : 'WS10_10mAVG', 'model' : 'WS10_10mAVG'},
            {'name': "30公尺風速器", 'type': 'ws', 'column' : 'WS30_10mAVG', 'model' : 'WS30_10mAVG'},
            {'name': "50公尺風速器", 'type': 'ws', 'column' : 'WS50_10mAVG', 'model' : 'WS50_10mAVG'},
            {'name': "95公尺風速器", 'type': 'ws', 'column' : 'WS95A_10mAVG','model' : 'WS95A_10mAVG'},
            
            {'name': "10公尺風向器", 'type': 'wd', 'column' : 'WD10_10mAVG', 'model' : 'WD10_10mAVG_cos'},
            {'name': "30公尺風向器", 'type': 'wd', 'column' : 'WD30_10mAVG', 'model' : 'WD30_10mAVG_cos'},
            {'name': "50公尺風向器", 'type': 'wd', 'column' : 'WD50_10mAVG', 'model' : 'WD50_10mAVG_cos'},
            {'name': "95公尺風向器", 'type': 'wd', 'column' : 'WD95_10mAVG', 'model' : 'WD95_10mAVG_cos'}
        ]
        
        self._feature_type_title = {
            'ws': ' Wind Speed（風速計，單位：m/s = 公尺/秒）',
            'wd': ' Wind Direction（風向計，單位：度）'
        }
        
        # self.x_column_list = list(self.y_feature_name_maplist.keys())
        # self.y_feature_maplist = { self.y_feature_name_maplist[k]:  k for k in list(self.y_feature_name_maplist.keys())}
        
        ## 原始資料
        self.origin_data = self._getOriginData()
        #self.data = origin_data.copy()
        
        ## 測試期間
        self.test_period = 5*144
        
        ## 移動窗格大小
        # 144 => 1天
        self.moving_window_size = 3*144
        
        ## y 欄位
        self.x_column_list = []
        self.y_feature = self.feature_list[0]
        
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
        self.valid_origin_data = None
        self.valid_data = None
        self.val_y = None
        self.val_x = None
        
        ## 控制圖的寬度
        self._this_window_width = this_window_width
        
    def _getOriginData(self):
        try:
            ### For colab
            data = pd.read_csv('/content/fbc_demo/wind_demo.txt')
        except FileNotFoundError:
            ### For local 端
            data =  pd.read_csv('wind_demo.txt')
        except:
            print('讀取網路資源')
            data =  pd.read_csv(
                io.StringIO(
                    requests.get('https://recognise.trendlink.io/model/wind_demo.txt', verify=False).content.decode('utf-8')
                )
            )
        ## 風向計要做 cos 處理
        for col in data.columns:
            if "WD"in col:
                data[f"{col}_cos"] = data[col].apply(lambda x : np.cos(x*np.pi/90))
        data['recordTime'] = data['recordTime'].apply(lambda x :datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
        data.index = data['recordTime']
        return data
            
    def _buildModelDataset(self):
        data = self.origin_data.copy()

        ##模型輸入特徵lag處理
        self.x_column_list = []
        for lag in range(1, self.moving_window_size+1):
            _x_lag = "%s_%d"%(self.y_feature['model'], lag)
            data[_x_lag] = data[self.y_feature['model']].shift(lag)
            self.x_column_list.append(_x_lag)
                
        data = data.dropna()
        self.train_data = data.iloc[:-self.test_period,:]
        self.test_data = data.iloc[-self.test_period:,:]
        
        self.train_y = self.train_data[[self.y_feature['model']]]
        self.train_x = self.train_data[self.x_column_list]
        
        self.test_y  = self.test_data[[self.y_feature['model']]]
        self.test_x  = self.test_data[self.x_column_list]

    def _trainModel(self):
        self._model = DecisionTreeRegressor(max_depth=self.tree_max_depth)
        self._model.fit(
            self.train_x.values, 
            self.train_y.values
        )
    ### 畫圖系列  
    def plotData(self, num= 5*144, data_type = 'ws'):
        display_data = self.origin_data.head(num).copy()
        _feature_list = [f for f in self.feature_list if f['type'] == data_type]
        _init_feature = _feature_list[0]
        
        buttons_list = []
        for idx, x_col in enumerate(_feature_list):
            _visible = [False for _ in _feature_list]
            _visible[idx] = True
            buttons_list.append({
                'label':  x_col['name'],
                "method":"update",
                "args":[
                    {"visible": _visible},
                    {"title": x_col['name']+ self._feature_type_title[data_type]}],
            })

        plot = go.Figure(
            data=[
                go.Scatter(
                    x=display_data.index, 
                    y=display_data[x_col['column']], 
                    name=x_col['name'], 
                    mode='lines+markers',
                    visible=False if x_col['column'] != _init_feature['column'] else True
                ) for x_col in _feature_list]
        )

        # Add dropdown
        plot.update_layout(
            title=_init_feature['name']+ self._feature_type_title[data_type],
            updatemenus=[dict(
                type="buttons",
                direction="up",
                buttons=buttons_list
        )],
                xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        plot.show()
        
    def _plotModelPrediction(self, display_data, the_y_feature):
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]]
        )
        
        _table_header = ['recordTime', the_y_feature['name']]
        _table_value =[
            display_data['recordTime'].tolist(),
            display_data[the_y_feature['model']].tolist(),
        ]
        
        if not self._model is None:
            # pred Y
            fig.add_trace(
                go.Scatter(
                    x=display_data['recordTime'], 
                    y=display_data['pred'], 
                    name= f'預測的%s'%(the_y_feature['name']), 
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
                x=display_data['recordTime'], 
                y=display_data[the_y_feature['model']], 
                name=f'真實的%s'%(the_y_feature['name']), 
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
        
        fig.update_layout(
            width=self._this_window_width,
            showlegend=True,
            title_text="%s-預測 (%s)"%(the_y_feature['name'], self._feature_type_title[the_y_feature['type']]),
            legend=dict(y=0.5, traceorder='reversed', font_size=16)
        )
        fig.show()   
    
    def plotPredictionTestData(self): 
        display_data = self.test_data.copy()
        if not 'recordTime' in display_data.columns:
            display_data['recordTime'] = range(1, display_data.shape[0]+1)
        if not self._model is None:
            test_prediction = self._model.predict(self.test_x.values)
            display_data['pred'] = test_prediction
            
        self._plotModelPrediction(display_data, self.y_feature)    
  
    def plotPredictionValidData(self): 
        display_data = self.valid_data.copy()
        if not 'recordTime' in display_data.columns:
            display_data['recordTime'] = range(1, display_data.shape[0]+1)
        if not self._model is None:
            test_prediction = self._model.predict(self.val_x.values)
            display_data['pred'] = test_prediction
            
        self._plotModelPrediction(display_data, self.y_feature)  
        
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
    def makeValidData(self, start_v, end_v):
        self.valid_origin_data = np.array([uniform(start_v, end_v) for _ in range(self.moving_window_size+ self.test_period)])

    def buildValidDataset(self):
        data = pd.DataFrame(self.valid_origin_data, columns=[self.y_feature['model']])
        
        ##模型輸入特徵lag處理
        self.x_column_list = []
        for lag in range(1, self.moving_window_size+1):
            _x_lag = "%s_%d"%(self.y_feature['model'], lag)
            data[_x_lag] = data[self.y_feature['model']].shift(lag)
            self.x_column_list.append(_x_lag)
    
        data = data.dropna()
        self.valid_data = data.copy()
        self.val_y  = self.valid_data [[self.y_feature['model']]]
        self.val_x  = self.valid_data [self.x_column_list]

    ### UI event handle
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
            # print('change : ', change)
            for f in self.feature_list:
                if f['name'] == change['new']:
                    self.y_feature =  f
                    self._model = None
                    break
            
    def dataProcessingButtonOnClick(self, button_event):
        self._buildModelDataset()
        
    def modelPredictionButtonOnClick(self, button_event):
        self._buildModelDataset()
        self._trainModel()
        
class DisplayManager(object):
    def __init__(self, output, content_manager: ContentManager):
        self._output = output
        self._content_manager = content_manager
        
        
        self._show_ws_button = widgets.Button(description="風速資料")
        self._show_wd_button = widgets.Button(description="風向資料")
        self._show_ws_button.on_click(self._showWSButtonOnClick)
        self._show_wd_button.on_click(self._showWDButtonOnClick)
        
        self._test_period_widget = widgets.IntText(
            value=self._content_manager.test_period,
            disabled=False
        )
        self._moving_window_size_widget = widgets.IntText(
            value=self._content_manager.moving_window_size,
            disabled=False
        )
        self._feature_column_widget = widgets.RadioButtons(
            options=[f['name'] for f in self._content_manager.feature_list],
            value=self._content_manager.y_feature['name'],
            layout={'width': 'max-content'}
        )

        self._real_prediction_button = widgets.Button(description="真實資料預測")
        self._valid_prediction_button = widgets.Button(description="模擬資料預測")
        
        self._tree_max_depth_widget = widgets.IntText(
            value=self._content_manager.tree_max_depth,
            disabled=False
        )
        
        ## observe event
        self._test_period_widget.observe(self._content_manager.testPeriodOnChange)
        self._moving_window_size_widget.observe(self._content_manager.movingWindowSizeOnChange)
        self._feature_column_widget.observe(self._yColumnOnChange)
        self._tree_max_depth_widget.observe(self._content_manager.treeMaxDepthOnChange)        
        self._real_prediction_button.on_click(self._realPredictionButtonOnClick)
        self._valid_prediction_button.on_click(self._validPredictionButtonOnClick)
        
        
        ## valid data
        
        # self._y_fr = self._makeFloatRangeSlider(self._content_manager.y_column)
        # self._y_fr.observe(self._handleSliderChange)
        self._valid_fr = self._makeFloatRangeSlider(self._content_manager.y_feature)
        self._valid_fr.observe(self._handleSliderChange)
            
    
    def _makeFloatRangeSlider(self, the_feature):
        col_name_value = self._content_manager.train_data[the_feature['model']]
        
        q_25 = col_name_value.quantile(0.25)
        q_75 = col_name_value.quantile(0.75)
        max_v = col_name_value.max()
        min_v = col_name_value.min()
        
        float_range_slider = widgets.FloatRangeSlider(
            value=[q_25, q_75],
            min= min_v,
            max= max_v,
            step=(max_v-min_v)/1000,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            # layout={'width': 'max-content'}
        )
        
        self._content_manager.makeValidData(q_25, q_75)
        return float_range_slider
    
    def _handleSliderChange(self, change):
        if change['name'] == '_property_lock' and len(change['new'].keys()) == 0:
            owner = change['owner']

    def _testPeriodOnChange(self, change):
        if change['name'] == 'value':
            self._content_manager.content_manager(change)
        
        
    def _showWSButtonOnClick(self, change):
        clear_output()
        self.displayDataDashboard()
        self._content_manager.plotData(data_type='ws')
        
        
    def _showWDButtonOnClick(self, change):
        clear_output()
        self.displayDataDashboard()
        self._content_manager.plotData(data_type='wd')
        
        
    def _realPredictionButtonOnClick(self, change):
        self._content_manager.modelPredictionButtonOnClick(change)
        clear_output()
        self.displayHyperParamDashboard()
        self._content_manager.plotPredictionTestData()
        self._content_manager.plotTree()
        self._content_manager.plotFeatureImportances()
    def _yColumnOnChange(self, change):
        if change['name'] == 'value':
            self._content_manager.yColumnOnChange(change)
            self._valid_fr = self._makeFloatRangeSlider(self._content_manager.y_feature)
            self._valid_fr.observe(self._handleSliderChange)
            # self._y_fr = self._makeFloatRangeSlider(self._content_manager.y_column)
            # self._y_fr.observe(self._handleSliderChange)
        
    def displayDataDashboard(self):
        display(self._show_ws_button, self._output)
        display(self._show_wd_button, self._output)
        
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
                self._feature_column_widget
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
        display(widgets.Box(
            [
                widgets.Label(value=self._content_manager.y_feature['name']),
                self._valid_fr
            ]
        ), self._output)
        display(self._valid_prediction_button, self._output)
            
    def _validPredictionButtonOnClick(self, b):
        if self._content_manager._model is None:
            clear_output()
            self.displayValidDashboard()
            print("請先訓練模型")
        else:
            ### 重build valid data
            self._content_manager.makeValidData(
                self._valid_fr.value[0],
                self._valid_fr.value[1]
            )
            self._content_manager.buildValidDataset()
            clear_output()
            self.displayValidDashboard()
            self._content_manager.plotPredictionValidData()


WINDOW_WIDTH_QUERY = """
    var w = window.screen.width;
    IPython.notebook.kernel.execute("this_window_width="+w);
    """
def getWindowWidth():
    Javascript(WINDOW_WIDTH_QUERY)
    if not 'this_window_width' in dir():
        this_window_width = 1920
    return this_window_width
        
    
    
