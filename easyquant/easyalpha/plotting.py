import plotly.express as px 
from plotly.figure_factory import create_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser






def summary_plot(report,excess=True,mode='notebook'):
    """
    """

    excess_str = 'excess' if excess else ''
    factor = report['factor']
    fig_list = []
    # ic_decay
    fig_ic_decay = px.bar(report['ic_decay'].mean(),title=f"{factor} ic decay  <br />universe:{report['universe']}")
    ic_decay_min = report['ic_decay'].mean().min()
    ic_decay_max = report['ic_decay'].mean().max()
    fig_ic_decay.update_layout(yaxis_range=[max(-0.2,ic_decay_min),min(0.2,ic_decay_max)])
    fig_list.append(fig_ic_decay)


    # ic-series
    ic_series = report['ic_series']
    ic_series_ma = ic_series.rolling(12).mean()
    fig_ic_series = make_subplots()
    index = ic_series_ma.index
    fig_ic_series.add_trace(
        go.Scatter(x=index ,y=ic_series_ma.values.tolist(),name='ic_ma12')
    )
    fig_ic_series.add_trace(
        go.Bar(x=index, y=ic_series.values.tolist(), name="ic")
    )
    fig_ic_series.update_layout(title=f"{factor}  ic series  <br />universe:{report['universe']}")
    fig_list.append(fig_ic_series)


    # 分组净值
    for _group in report['groups'].keys():
        sub_report = report['groups'][_group]
        group_nvs = sub_report['group_nvs']
        fig_group_nvs = px.line(group_nvs-1)
        fig_group_nvs.update_layout(title=f"{factor}  cumulative {excess_str} return(compound) of different groups <br />groups:{_group} | universe:{report['universe']}")
        # fig_group_nvs.show()
        fig_list.append(fig_group_nvs)

        
    for _group in report['groups'].keys():
        # 多头收益
        sub_report = report['groups'][_group]
        group_return_long = sub_report['group_return_long']
        group_return_long_ma = group_return_long.rolling(12).mean()
        fig_group_return_long = make_subplots()
        index = group_return_long_ma.index
        fig_group_return_long.add_trace(
            go.Scatter(x=index ,y=group_return_long_ma.values.tolist(),name='return_ma12')
        )
        fig_group_return_long.add_trace(
            go.Bar(x=index, y=group_return_long.values.tolist(), name="return")
        )
        fig_group_return_long.update_layout(title=f"{factor}   long side {excess_str} return<br />groups:{_group} | universe:{report['universe']}")
        # fig_group_return_long.show()
        fig_list.append(fig_group_return_long)

    # for _group in report['groups'].keys():
        # 空头收益
        sub_report = report['groups'][_group]
        group_return_short = sub_report['group_return_short']
        group_return_short_ma = group_return_short.rolling(12).mean()
        fig_group_return_short = make_subplots()
        index = group_return_short_ma.index
        fig_group_return_short.add_trace(
            go.Scatter(x=index ,y=group_return_short_ma.values.tolist(),name='return_ma12')
        )
        fig_group_return_short.add_trace(
            go.Bar(x=index, y=group_return_short.values.tolist(), name="return")
        )
        fig_group_return_short.update_layout(title=f"{factor}   short side {excess_str} return<br />groups:{_group} | universe:{report['universe']}")
        # fig_group_return_short.show()
        fig_list.append(fig_group_return_short)

    for _group in report['groups'].keys():
        sub_report = report['groups'][_group]    
        fig_ret_SR_TO = make_subplots(rows=1,cols=3)
        # 分组年化收益
        ann_ret = sub_report['ann_ret']
        index = ann_ret.index
        fig_ret_SR_TO.add_trace(
            go.Bar(x=index,y=ann_ret.values,name=f'annual {excess_str} return'),row=1,col=1
        )
        # fig_ann_ret.show()

        # 分组夏普
        SR = sub_report['SR']
        index = SR.index
        fig_ret_SR_TO.add_trace(
            go.Bar(x=index,y=SR.values,name=f'SR'),row=1,col=2
        )

        # 分组换手
        TO = sub_report['TO']
        index = TO.index
        fig_ret_SR_TO.add_trace(
            go.Bar(x=index,y=TO.values,name='TO'),row=1,col=3
        )
        fig_ret_SR_TO.update_layout(title=f"{factor}   annual {excess_str} return, SR and TO<br />groups:{_group} | universe:{report['universe']}")
        fig_list.append(fig_ret_SR_TO)

    for _group in report['groups'].keys():
        # 分组评估指标
        sub_report = report['groups'][_group]
        perfs = sub_report['excess_performance']
        perfs = perfs.rename(index={'turnover_ratio':'TO'})
        table_perfs = create_table(perfs.reset_index())
        table_perfs.update_layout(title=f"{excess_str} performance indicator  <br />groups:{_group} | universe:{report['universe']}")
        fig_list.append(table_perfs)
        # for _fig in fig_list:


    # bar_factor_ana = px.bar(report['factor_ana'],x='set',y='ic.mean',color='group',barmode='group',title='ic of different ep & liquidity')
    # bar_factor_ana.update_layout(title=f"{factor}  ic of different sort<br />universe:{report['universe']}")
    # fig_list.append(bar_factor_ana)

    for _i in range(len(fig_list)):
        _fig = fig_list[_i]
        if mode !='notebook':
            mode = 'a' if _i > 0 else 'w'
            file = factor.replace(' ',"_")+'.html'
            with open(file, mode) as f:    
                f.write(_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        else:
            _fig.show()
    if mode !='notebook':
        webbrowser.open( file,new=2)



