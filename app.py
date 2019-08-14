from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle
import dill
from sklearn import base

import bokeh.plotting as bk
import bokeh.embed as bke
import folium


#MAPBOX_APIKEY = 'pk.eyJ1IjoiYWx5Yml0dCIsImEiOiJjanlodGRzaGQwMzk4M2RwMHBnbG91bjk0In0.PbpJZgQlO_2Ce1JS-fXIKw'
    
#with open ('/Users/Aly/gz_2010_us_050_00_20m.json', encoding = "ISO-8859-1") as fr:
#    county_js = json.load(fr)
    
county_js = dill.load(open('static/county_js_ny.pkd','rb'))
data_adjust_vax = pd.read_csv('static/data_adjust_vax.csv')
data_by_county = pd.read_csv('static/data.csv')

def get_new_rows(df):
    new_rows = []
    for row in df.values:
        for i in range(101):
            new_row = row.copy()
            new_row[3] = i*1000
            new_row[-1] = 0
            new_rows.append(new_row)
    return new_rows

def get_cutoff_plot(data_by_county):
    sort_cases = data_by_county.sort_values('Avg Cases')
    risk_0 = sort_cases[sort_cases['Avg Cases'] == 0]
    risk_1 = sort_cases[(sort_cases['Avg Cases'] > 0) & ((sort_cases['Avg Cases'] < 1.9))]
    risk_2 = sort_cases[(sort_cases['Avg Cases'] >=1.9) & ((sort_cases['Avg Cases'] < 7))]
    risk_3 = sort_cases[sort_cases['Avg Cases'] >= 7]

    l_0 = len(risk_0['Avg Cases'].values)
    l_1 = len(risk_1['Avg Cases'].values)
    l_2 = len(risk_2['Avg Cases'].values)
    l_3 = len(risk_3['Avg Cases'].values)

    p = bk.figure(title = 'Number of Cases of Vaccine Preventable Diseases per 100,000',plot_width = 800, plot_height = 400,background_fill_alpha = 0.75)
    p.circle(range(l_0), risk_0['Avg Cases'].values, color = 'green', legend = 'risk level 0')
    p.circle(range(l_0, l_0+l_1), risk_1['Avg Cases'].values, color = '#FEE300', legend = 'risk level 1')
    p.circle(range(l_0+l_1, l_0+l_1+l_2), risk_2['Avg Cases'].values, color = 'orange', legend = 'risk level 2')
    p.circle(range(l_0+l_1+l_2,l_0+l_1+l_2+l_3) , risk_3['Avg Cases'].values, color = 'red', legend = 'risk level 3')
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.75
    p.border_fill_alpha = 0.75
    return p
    
def get_risk_plot(county):
    df_line_plot = new_df[new_df['County'] == county].sort_values('Known Unvax per 100,000')
    p = bk.figure(title = 'risk plot for '+county.lower()+' county',plot_width = 600, plot_height = 300)
    p.xaxis.axis_label = 'Unvax Per 100,000'
    p.yaxis.axis_label = 'Risk Level'    
    p.line(df_line_plot['Known Unvax per 100,000'].values, df_line_plot['Pred'], color = 'red')
    p.circle(df_line_plot['Known Unvax per 100,000'].values, df_line_plot['Pred'], color = 'red', size = 2.5)
    p.background_fill_alpha = 0.75
    p.border_fill_alpha = 0.75
    return p
    

def get_folium_plot(data):
    m = folium.Map(location=[42.75, -76], zoom_start=7)

    choropleth = folium.Choropleth(
        geo_data=county_js,
        name='choropleth',
        data=data,
        columns=['GEO_ID', 'Pred'],
        key_on='feature.properties.GEO_ID',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color = 'white',
        nan_fill_opacity = 0,
        legend_name= 'Risk Level'
    ).add_to(m)
    
    choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['County', 'Unvax per 100,000'])
)
    return m
    
def get_plot_dict(data, model):
    d = {}
    data = data.set_index('County')
    for county in data.index:
        d[county] = {}
        for i in range(11):
            df = data.copy()
            key = i*10000
            df.loc[county, 'Known Unvax per 100,000'] = key
            df['Pred'] = model.predict(np.array(df[['Ratio Int Travelers', 'Known Unvax per 100,000', 'Population Density','Latitude','Longitude']]))
            d[county][key] = df.reset_index()
            
    county_plots = {}
    for county in d.keys():
        county_plots[county] = {}
        for i in d[county].keys():
            df = d[county][i]
            county_plots[county][i] = get_folium_plot(df)
            
    return county_plots
    
class stack_estimators(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        
    def fit(self,X,y):
        X2, y2 = ros.fit_resample(X, y) #balance
        self.estimator1 = self.estimator1.fit(X2, y2)
        
        X3 = np.vstack((X.T, self.estimator1.predict(X))).T
        X4, y4 = ros.fit_resample(X3,y)
        
        self.estimator2 = self.estimator2.fit(X4,y4)
        
        return self
    
    def predict(self, X):
        X2 = np.vstack((X.T, self.estimator1.predict(X))).T
        return self.estimator2.predict(X2)
    
model = pickle.load(open('static/finalized_model.sav', 'rb'))

class_data = pd.read_csv('static/class_data.csv')
class_data['Pred'] = class_data['Risk Level']

new_df = pd.DataFrame(get_new_rows(data_adjust_vax), columns = list(data_adjust_vax.columns))
new_df['Pred'] = model.predict(np.array(new_df[['Ratio Int Travelers', 'Known Unvax per 100,000', 'Population Density','Latitude','Longitude']]))

cutoff_plot = get_cutoff_plot(data_by_county)
orig_plot = get_folium_plot(class_data)
plot_dict = get_plot_dict(data_adjust_vax, model)


app = Flask(__name__)

@app.route('/')
def index():
    folium_map = orig_plot
    folium_map.save('static/htmls/map.html')
    cutoff_plot = get_cutoff_plot(data_by_county)
    script,div = bke.components(cutoff_plot)
    risk_map = get_risk_plot('ROCKLAND')
    script2,div2 = bke.components(risk_map)
    return render_template('index_folium.html', script = script, div = div, script2 = script2, div2 = div2)
    #return render_template('test.html', script = script, div = div, script2 = script2, div2 = div2)

    
@app.route('/interactive_plot', methods = ['GET', 'POST'])
def interactive_plot():
    if request.method == 'GET':
        folium_map = orig_plot
        risk_map = get_risk_plot('ALBANY')
    else:
        c = str(request.form.get("county", None))
        u = int(request.form.get("unvax", None))
        folium_map = plot_dict[c][u]
        risk_map = get_risk_plot(c)
    
    folium_map.save('static/htmls/map.html')
    script,div = bke.components(cutoff_plot)
    script2,div2 = bke.components(risk_map)
    return render_template('index_folium.html', script = script, div = div, script2 = script2, div2 = div2)
    #return render_template('test.html', script = script, div = div, script2 = script2, div2 = div2)

if __name__ == '__main__':
    app.run(port=33507, debug = True)
    #app.run()
