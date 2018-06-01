Viridis256 = [
    '#440154', '#440255', '#440357', '#450558', '#45065A', '#45085B', '#46095C', '#460B5E', '#460C5F', '#460E61',
    '#470F62', '#471163',
    '#471265', '#471466', '#471567', '#471669', '#47186A', '#48196B', '#481A6C', '#481C6E', '#481D6F', '#481E70',
    '#482071', '#482172',
    '#482273', '#482374', '#472575', '#472676', '#472777', '#472878', '#472A79', '#472B7A', '#472C7B', '#462D7C',
    '#462F7C', '#46307D',
    '#46317E', '#45327F', '#45347F', '#453580', '#453681', '#443781', '#443982', '#433A83', '#433B83', '#433C84',
    '#423D84', '#423E85',
    '#424085', '#414186', '#414286', '#404387', '#404487', '#3F4587', '#3F4788', '#3E4888', '#3E4989', '#3D4A89',
    '#3D4B89', '#3D4C89',
    '#3C4D8A', '#3C4E8A', '#3B508A', '#3B518A', '#3A528B', '#3A538B', '#39548B', '#39558B', '#38568B', '#38578C',
    '#37588C', '#37598C',
    '#365A8C', '#365B8C', '#355C8C', '#355D8C', '#345E8D', '#345F8D', '#33608D', '#33618D', '#32628D', '#32638D',
    '#31648D', '#31658D',
    '#31668D', '#30678D', '#30688D', '#2F698D', '#2F6A8D', '#2E6B8E', '#2E6C8E', '#2E6D8E', '#2D6E8E', '#2D6F8E',
    '#2C708E', '#2C718E',
    '#2C728E', '#2B738E', '#2B748E', '#2A758E', '#2A768E', '#2A778E', '#29788E', '#29798E', '#287A8E', '#287A8E',
    '#287B8E', '#277C8E',
    '#277D8E', '#277E8E', '#267F8E', '#26808E', '#26818E', '#25828E', '#25838D', '#24848D', '#24858D', '#24868D',
    '#23878D', '#23888D',
    '#23898D', '#22898D', '#228A8D', '#228B8D', '#218C8D', '#218D8C', '#218E8C', '#208F8C', '#20908C', '#20918C',
    '#1F928C', '#1F938B',
    '#1F948B', '#1F958B', '#1F968B', '#1E978A', '#1E988A', '#1E998A', '#1E998A', '#1E9A89', '#1E9B89', '#1E9C89',
    '#1E9D88', '#1E9E88',
    '#1E9F88', '#1EA087', '#1FA187', '#1FA286', '#1FA386', '#20A485', '#20A585', '#21A685', '#21A784', '#22A784',
    '#23A883', '#23A982',
    '#24AA82', '#25AB81', '#26AC81', '#27AD80', '#28AE7F', '#29AF7F', '#2AB07E', '#2BB17D', '#2CB17D', '#2EB27C',
    '#2FB37B', '#30B47A',
    '#32B57A', '#33B679', '#35B778', '#36B877', '#38B976', '#39B976', '#3BBA75', '#3DBB74', '#3EBC73', '#40BD72',
    '#42BE71', '#44BE70',
    '#45BF6F', '#47C06E', '#49C16D', '#4BC26C', '#4DC26B', '#4FC369', '#51C468', '#53C567', '#55C666', '#57C665',
    '#59C764', '#5BC862',
    '#5EC961', '#60C960', '#62CA5F', '#64CB5D', '#67CC5C', '#69CC5B', '#6BCD59', '#6DCE58', '#70CE56', '#72CF55',
    '#74D054', '#77D052',
    '#79D151', '#7CD24F', '#7ED24E', '#81D34C', '#83D34B', '#86D449', '#88D547', '#8BD546', '#8DD644', '#90D643',
    '#92D741', '#95D73F',
    '#97D83E', '#9AD83C', '#9DD93A', '#9FD938', '#A2DA37', '#A5DA35', '#A7DB33', '#AADB32', '#ADDC30', '#AFDC2E',
    '#B2DD2C', '#B5DD2B',
    '#B7DD29', '#BADE27', '#BDDE26', '#BFDF24', '#C2DF22', '#C5DF21', '#C7E01F', '#CAE01E', '#CDE01D', '#CFE11C',
    '#D2E11B', '#D4E11A',
    '#D7E219', '#DAE218', '#DCE218', '#DFE318', '#E1E318', '#E4E318', '#E7E419', '#E9E419', '#ECE41A', '#EEE51B',
    '#F1E51C', '#F3E51E',
    '#F6E61F', '#F8E621', '#FAE622', '#FDE724']

import sys
import textwrap

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, widgetbox, layout, row
from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    Slider,
    RadioButtonGroup,
    PreText
)
from bokeh.plotting import figure

# from bokeh.sampledata.unemployment1948 import data

sys.path.append('/mnt/data/tommy8054/MovieQA_Contest')
from config import MovieQAPath
from data.data_loader import QA, Subtitle

curdoc().clear()
# data['Year'] = data['Year'].astype(str)
# data = data.set_index('Year')
# data.drop('Annual', axis=1, inplace=True)
# data.columns.name = 'Month'
#
# years = list(data.index)
# months = list(data.columns)
# df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()

_mp = MovieQAPath()

train_qa = QA().include(video_clips=True, split={'train'}).get()
val_qa = QA().include(video_clips=True, split={'val'}).get()
qa = [train_qa, val_qa]
subtitle = Subtitle().get()

# ins = train_qa[0]
# attn = np.eye(len(subtitle[ins['imdb_key']]['lines']), 7)
# data = pd.DataFrame(
#     data=attn,
#     # columns=[str(i) for i in range(7)],
#     columns=[ins['question'], 'belief'] + ins['answers'],
#     # index=[str(i) for i in range(len(subtitle[ins['imdb_key']]['lines']))]
#     index=['%d. ' % idx + l for idx, l in enumerate(subtitle[ins['imdb_key']]['lines'])]
# )
#
# data.index.name = 'subt'
# data.columns.name = 'attn'
# df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()
# source = ColumnDataSource(df)

source = ColumnDataSource(data=dict(attn=[], subt=[], rate=[], index=[]))
mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
p = figure(  # title="Attention map",
    x_range=[str(i) for i in range(7)],
    # x_range=list(data.columns),
    y_range=[str(i) for i in range(100)],
    # y_range=list(reversed(list(data.index))),
    y_axis_location="right",
    x_axis_location="above", plot_width=700, plot_height=700,
    tools='', toolbar_location=None, output_backend="webgl")

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "8pt"
p.axis.major_label_standoff = 0
# p.xaxis.major_label_orientation = pi / 3

p.rect(x="attn", y="subt", width=1, height=1,
       source=source,
       fill_color={'field': 'rate', 'transform': mapper},
       line_color=None)

# color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
#                      label_standoff=6, border_line_color=None, location=(0, 0))
# p.add_layout(color_bar, 'left')

# output_file('./visualize/test.html')
# div = PreText(text='\n'.join([train_qa[0]['question']] +
#                              train_qa[0]['answers'] +
#                              subtitle[train_qa[0]['imdb_key']]['lines']),
#               height=600, width=1200)
box = column(p, height=700, width=700, sizing_mode='fixed', css_classes=['scrollable'])
div = PreText(text='0.0',
              height=600, width=600)
box = row([box, div])


def update():
    # div.text = '\n'.join([train_qa[qa_slider.value]['question']] +
    #                      train_qa[qa_slider.value]['answers'] +
    #                      subtitle[train_qa[qa_slider.value]['imdb_key']]['lines'])

    new_ins = qa[button.active][qa_slider.value]
    num_subt = len(subtitle[new_ins['imdb_key']]['lines'])
    new_attn = np.random.random((num_subt, 7))
    new_data = pd.DataFrame(
        data=new_attn,
        columns=[str(i) for i in range(7)],
        index=['%d. ' % idx + l for idx, l in enumerate(subtitle[new_ins['imdb_key']]['lines'])])
    new_data.index.name = 'subt'
    new_data.columns.name = 'attn'
    new_df = pd.DataFrame(new_data.stack(), columns=['rate']).reset_index()
    source.data = dict(attn=new_df['attn'], subt=new_df['subt'], rate=new_df['rate'], index=new_df.index)
    p.x_range.factors = list(new_data.columns)
    p.y_range.factors = list(reversed(new_data.index))
    p.height = num_subt * 12
    # p.plot_height = num_subt * 12
    div.text = '\n'.join([textwrap.fill('Q: ' + new_ins['question'], 70)] +
                         [textwrap.fill(('%d: ' % idx) + ans, 70) for idx, ans in enumerate(new_ins['answers'])])


def b_update():
    qa_slider.end = len(qa[button.active]) - 1
    qa_slider.value = 0
    update()


qa_slider = Slider(start=0, end=len(train_qa) - 1, value=0, step=1, title='qa #',
                   width=1200)
qa_slider.on_change('value', lambda attr, old, new: update())
button = RadioButtonGroup(labels=['train', 'validation'], active=0)
button.on_change('active', lambda attr, old, new: b_update())
control = widgetbox(button, qa_slider)

curdoc().add_root(layout([[control], [box]]))
# curdoc().add_root(control)
# curdoc().add_root(box)
curdoc().title = "Visualize"

update()
