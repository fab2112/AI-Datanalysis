
import pandas as pd
import plotly.express as px

df = pd.DataFrame(DF_1)

fig = px.bar(df, x="diagnosis", y="radius_mean", title="Radius Mean by Diagnosis")

fig.update_layout(width=800,
                  height=600,
                  plot_bgcolor=None,
                  paper_bgcolor=None,
                  clickmode='event+select',
                  margin=dict(l=40, r=40, t=65, b=65),
                  font=dict(family="Open Sans, sans-serif"),
                  xaxis=dict(showgrid=True,
                             tickfont=dict(size=14),
                             showline=True,
                             linewidth=1,
                             linecolor='grey',
                             mirror=True),
                  yaxis=dict(showgrid=True,
                             tickfont=dict(size=14),
                             showline=True,
                             linewidth=1,
                             linecolor='grey',
                             mirror=True),
                  title=dict(font=dict(family="Open Sans, sans-serif"), x=0.08, font_size=20),
                  xaxis_title=dict(text="Diagnosis", font=dict(size=14)),
                  yaxis_title=dict(text="Radius Mean", font=dict(size=14)),
                  legend_title=dict(text=""))

fig.update_traces(marker=dict(line_width=0.7, line_color='#E8E8E8', opacity=1, cornerradius=4),
                  hoverlabel=dict(bgcolor='#383838', font=dict(color='white')),
                  hovertemplate="<b>Diagnosis</b>: %{x}<br><b>Radius Mean</b>: %{y}<extra></extra>")

result = fig
