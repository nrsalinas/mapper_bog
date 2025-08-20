# coding: utf-8

import streamlit as st
import io
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from geographiclib.geodesic import Geodesic

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
from matplotlib_map_utils import set_size

set_size("small")

###########################   User variables   #################################

BOG_SHAPE = "localidades_bogota/Loca.shp"

#? include cmap option
#? coolwarm, jet, viridis

###########################   Functions.       #################################

def insert_arrow(axis, x_pos=0.95, y_pos=0.08, size=0.06, width=3, headwidth=8,
	fontsize=10):
	"""
	Inserts a N arrow into a map. Position is set with parameters `x_pos` and 
	`y_pos`, which indicate fractions along the axes, from left to right or from 
	top to bottom.
	"""
	x, y, arrow_length = x_pos, y_pos, size
	axis.annotate(
		'N', 
		xy=(x, y), 
		xytext=(x, y-arrow_length),
		arrowprops=dict(facecolor='black', width=width, headwidth=headwidth),
		ha='center', 
		va='center', 
		fontsize=fontsize,
		xycoords=axis.transAxes
	)


def insert_scale(axis, lat1, lon1, lat2, lon2, position = "upper right"):
	"""
	
	The simplest scale on earth.

	- axis: A matplotlib ax object.

	- lat1, lon1, lat2, lon2: Coordinates of a line that has the same size as 
	the scale. 
	
	- position: `upper right`, `upper left`, `lower right`, or `lower left`.
	
	"""
	
	pp = gpd.GeoSeries(
		[
			shapely.geometry.point.Point(lon1, lat1), 
			shapely.geometry.point.Point(lon2, lat2)
		], 
		crs=4326
	)

	scale_meters = pp[0].distance(pp[1])
	axis.add_artist(
		ScaleBar(
			scale_meters,
			location=position, #"upper right"
			scale_formatter = lambda value, unit: f"",
			border_pad = 1,
		)
	)

	return None


def deserves_inset(
		data_points: gpd.GeoDataFrame, 
		region: gpd.GeoDataFrame, 
		coverage:float = 0.5
		) -> bool:
	""""
	Decide if a map deserves an inset of the region.
	
	- data_points: Occurrences, using DarwinCore standard.

	- region: Shapefile of the region.

	- coverage: Fraction (0-1) of the region extent covered by the 
	occurrences extent. Occurrences extent is the area of the rectangle of the 
	latitudinal and longitudinal span of the data. Region extent is the area of 
	the rectangle of the latitudinal and longitudinal span of the region.
	
	"""
	lon_min, lat_min, lon_max, lat_max = data_points.dissolve(
		).bounds.values.flatten().tolist()
	
	area_data = (lat_max - lat_min) * (lon_max - lon_min) 
	
	lon_min, lat_min, lon_max, lat_max = region.dissolve(
		).bounds.values.flatten().tolist()
	
	area_region = (lat_max - lat_min) * (lon_max - lon_min) 

	if area_data / area_region > coverage:
		return False
	else:
		return True

def remove_uncert(records, region, centroid_buffer):
	"""
	Drop records with uncertain locality information that were geocoded using 
	centroids of a given region.

	records (GeoDataFrame): Records following the DarwinCore standard. Locality
	information should be inserted in the `verbatimLocality` column.

	region (GeoDataFrame): Shapefile of the area of interest.

	centroid_buffer (float): Buffer to use around the region centroid to select 
	records as candidates to remove. Should be the same spatial unit of `region`. 
	"""
	centr = region.dissolve().centroid
	to_drop = records.sjoin(
		gpd.GeoDataFrame(centr.buffer(centroid_buffer), geometry=0).to_crs(4326)
		).query('verbatimLocality.isna()').index
	return records.drop(index=to_drop)


def get_plot_extent(
	data_points: gpd.GeoDataFrame, 
	inset_pos: str = 'upper left',
	buffer_perc_inset: float = 0.2,
	buffer_perc_no_inset: float = 0.1,
	) -> tuple:
	"""
	Given a set of occurrences and a shapefile, retrieves the geographic limits
	to plot. 

	- data_points: Occurrences, using DarwinCore standard.

	- inset_pos: Position of the inset within the plot. Acepted values: 
	`upper left` (default), `upper right`, `bottom left`, and `bottom right`.

	- buffer_perc_inset: Buffer (as fraction) to be added to the plot extent on 
	the inset side. Default: 0.2.
	
	- buffer_perc_no_inset: Buffer (as fraction) to be added to the plot extent 
	on the side opposing the inset. Default: 0.1.
	"""

	bf_up = None
	bf_bo = None
	bf_lf = None
	bf_rg = None

	lon_min, lat_min, lon_max, lat_max = data_points.dissolve(
		).bounds.values.flatten().tolist()

	if inset_pos == 'upper left':
		bf_up = buffer_perc_inset
		bf_bo = buffer_perc_no_inset
		bf_lf = buffer_perc_inset
		bf_rg = buffer_perc_no_inset
		
	elif inset_pos == 'upper right':
		bf_up = buffer_perc_inset
		bf_bo = buffer_perc_no_inset
		bf_lf = buffer_perc_no_inset
		bf_rg = buffer_perc_inset
		
	elif inset_pos == 'bottom left':
		bf_up = buffer_perc_no_inset
		bf_bo = buffer_perc_inset
		bf_lf = buffer_perc_inset
		bf_rg = buffer_perc_no_inset
		
	elif inset_pos == 'bottom right':
		bf_up = buffer_perc_no_inset
		bf_bo = buffer_perc_inset
		bf_lf = buffer_perc_no_inset
		bf_rg = buffer_perc_inset

	bffr_lat_0 = (lat_max - lat_min) * bf_bo
	bffr_lat_1 = (lat_max - lat_min) * bf_up
	bffr_lon_0 = (lon_max - lon_min) * bf_lf
	bffr_lon_1 = (lon_max - lon_min) * bf_rg
	plot_ext_lat = (lat_min - bffr_lat_0), (lat_max + bffr_lat_1)
	plot_ext_lon = (lon_min - bffr_lon_0), (lon_max + bffr_lon_1)

	return (plot_ext_lat, plot_ext_lon)


def generate_maps():

	if st.session_state.token != st.secrets.token:
		st.session_state.error_message += "El token de autenticación es incorrecto.\n\n"

	if not st.session_state.outpoints:
		st.session_state.error_message += "Es necesario proporcionar un nombre de archivo para el mapa de puntos.\n\n"

	if st.session_state.scale_pos == st.session_state.inset_pos:
		st.session_state.error_message += "La ubicación de la escala no puede ser igual a la ubicación del inset.\n\n"

	if len(st.session_state.error_message) > 0:
		return None

	dist5km = Geodesic.WGS84.Direct(4.11, -74.11, 0, 5000)
	onekm = (((dist5km['lat1'] - dist5km['lat2'])**2 + (dist5km['lon1'] - dist5km['lon2'])**2)**0.5) / 1


	data = st.session_state.data
	data = data.loc[data.decimalLatitude.notna() & data.decimalLongitude.notna()
		].reset_index(drop=True)

	points = gpd.GeoDataFrame(
		data[
			['decimalLatitude', 'decimalLongitude', 'scientificName',
			'stateProvince', 'county', 'verbatimLocality']
		], 
		geometry=gpd.points_from_xy(data.decimalLongitude, data.decimalLatitude),
		crs=4326
		)

	frame = gpd.read_file(BOG_SHAPE)
	frame = frame.to_crs(4326)

	bbounds = pd.concat([
		frame.bounds.apply(min, axis=0).rename('min'), 
		frame.bounds.apply(max, axis=0).rename('max')
		], axis=1)

	points = remove_uncert(points, frame, onekm * 5)
	fpoints = gpd.sjoin(points, frame, how='inner', predicate='within')

	# Check plot extent

	plot_ext_lat, plot_ext_lon = None, None

	if deserves_inset(fpoints, frame):
		
		plot_ext_lat, plot_ext_lon = get_plot_extent(fpoints, st.session_state.inset_pos)

	######################    Plot first map    #####################################

	fig, ax = plt.subplots(figsize=(12, 8))

	if plot_ext_lat and plot_ext_lon:
		# Set map limits
		ax.set_xlim(plot_ext_lon)
		ax.set_ylim(plot_ext_lat)

		iax = inset_map(ax, location=st.session_state.inset_pos, size=2, xticks=[], yticks=[])
		frame.plot(ax=iax, linewidth=0.5)
		indicate_extent(pax=iax, bax=ax, pcrs=4326, bcrs=4326)


	frame.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)
	fpoints.plot(ax=ax, markersize=2)

	insert_scale(ax, dist5km['lat1'], dist5km['lon1'], dist5km['lat2'], 
		dist5km['lon2'], st.session_state.scale_pos)

	insert_arrow(ax)

	plt.title(st.session_state.title, pad=25)
	plt.xlabel('Longitud', labelpad=15)
	plt.ylabel('Latitud', labelpad=15)
	#plt.savefig(st.session_state.outpoints, dpi=300, bbox_inches='tight')
	plt.savefig(st.session_state.bffr0, dpi=300, bbox_inches='tight', format='png')
	st.session_state.dots_ready = True

	if plot_ext_lat is None and plot_ext_lon is None:

		##################    Check data     ###########################################

		if not st.session_state.heat_map_var in data.columns.tolist():
			st.session_state.error_message += "La variable categórica no corresponde a alguna de las columnas en la matriz de datos.\n\n"

		if not st.session_state.outheat:
			st.session_state.error_message += "El necesario proporcionar un nombre de archivo para el mapa de calor.\n\n"

		elif st.session_state.outheat == st.session_state.outpoints:
			st.session_state.error_message += "El nombre de archivo para el mapa de calor debe ser diferente al del archivo de mapa de puntos.\n\n"

		if len(st.session_state.error_message) > 0:
			error_window(st.session_state.error_message)
			st.session_state.error_message = ""
			return None		

		##################    Create grid for hotmap     ###############################
		
		cell_size = dist5km['a12'] / 5 * 2 # 2 km in coordinates
		mycrs = 4326 # projection of the grid
		grid_cells = []

		# create the cells in a loop
		for x0 in np.arange(bbounds.loc['minx','min'], bbounds.loc['maxx','max']+cell_size, cell_size ):

			for y0 in np.arange(bbounds.loc['miny','min'], bbounds.loc['maxy','max']+cell_size, cell_size):
			
				# bounds
				x1 = x0-cell_size
				y1 = y0+cell_size
				grid_cells.append(shapely.geometry.box(x0, y0, x1, y1)  )

		cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], 
				crs=mycrs)

		# Variable counts for heap map
		merged = gpd.sjoin(fpoints, cell, how='left', predicate='within',
			lsuffix='points', rsuffix='cells')
		dissolve = merged.loc[merged[st.session_state.heat_map_var].notna()
			].dissolve(
			by="index_cells",
			aggfunc={
				st.session_state.heat_map_var: (lambda i: np.unique(i))   
			}
		)
		dissolve['n_cats'] = dissolve[st.session_state.heat_map_var].apply(lambda m: m.shape[0])
		cell.loc[dissolve.index, 'n_cats'] = dissolve.n_cats.values
		thmax = cell.n_cats.max()


		###################      Plot second map     ###################################

		fig, ax = plt.subplots(figsize=(12, 8))

		if plot_ext_lat and plot_ext_lon:
			# Set map limits
			ax.set_xlim(plot_ext_lon)
			ax.set_ylim(plot_ext_lat)

			iax = inset_map(ax, location=st.session_state.inset_pos, size=2, xticks=[], yticks=[])
			frame.plot(ax=iax, linewidth=0.5)
			indicate_extent(pax=iax, bax=ax, pcrs=4326, bcrs=4326)
			
		frame.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)
		fpoints.plot(ax=ax, markersize=2)
		f0 = frame.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)
		f1 = cell.plot(ax=ax, column='n_cats', cmap='viridis', vmax=thmax, edgecolor="white", legend=True)

		insert_scale(ax, dist5km['lat1'], dist5km['lon1'], dist5km['lat2'], 
			dist5km['lon2'], st.session_state.scale_pos)

		insert_arrow(ax)


		plt.title(st.session_state.title, pad=25)
		plt.xlabel('Longitud', labelpad=15)
		plt.ylabel('Latitud', labelpad=15)
		#plt.savefig(st.session_state.outheat, dpi=300, bbox_inches='tight')
		plt.savefig(st.session_state.bffr1, dpi=300, bbox_inches='tight', format='png')
		st.session_state.heat_ready = True

@st.dialog("Error")
def error_window(message):
	st.write(message)

st.markdown("""

# Jardín Botánico de Bogotá

## Programa Conservación _in situ_

### Mapper: una aplicación simple de mapeo de registros.

#### Instrucciones

1. Cargue un archivo csv en formato DarwinCore con los registros biológicos que 
desea mapear. El archivo debe contener —mínimamente— las coordenadas geográficas 
en formato decimal, en columnas nombradas **decimalLatitude** y **decimalLongitude**. 
Si se quiere realizar un mapa de calor utilizando una característica para agregar 
valores, se debe indicar en la casilla **Variable categórica** el título de dicha 
columna (por ejemplo, **scientificName**). El archivo no puede superar las 200 MB.

2. Especifíque los parámetros del mapa, como el título de la gráfica, los nombres 
de los archivos de salida y el nombre de la variable para realizar el mapa de 
calor.
			
3. Presione el botón :red[**Enviar**].
			
4. Descargue los archivos.


----
						
""")

if not "data" in st.session_state: 
	st.session_state.data = None

if not "error_message" in st.session_state: 
	st.session_state.error_message = ""

if not "bffr0" in st.session_state:
	st.session_state.bffr0 = io.BytesIO()

if not "bffr1" in st.session_state:
	st.session_state.bffr1 = io.BytesIO()

if not "dots_ready" in st.session_state:
	st.session_state.dots_ready = False

if not "heat_ready" in st.session_state:
	st.session_state.heat_ready = False

with st.form(
	"Mapper - main",
	clear_on_submit=False,
	):

	st.text_input(
		"Token de autenticación",
		help="Token de validación de usuario",
		placeholder='Digite el token',
		value=None,
		key="token"
	)

	uploaded = st.file_uploader(
		"Archivo csv",
		type='csv',
		accept_multiple_files = False,
		key='intable',
		help='Archivo csv de registros en formato DarwinCore',
	)

	st.text_input(
		"Título",
		help="Título de los mapas",
		placeholder="Título de los mapas",
		value=None,
		key="title"
	)

	st.text_input(
		"Variable categórica",
		help="Variable categorica para agregar registros y proporcionar mapa de calor.",
		placeholder="Variable categórica",
		value="scientificName",
		key="heat_map_var"
	)

	st.text_input(
		"Nombre para el archivo de mapa de puntos",
		help="Nombre del archivo png que contendrá el mapa de puntos.",
		placeholder='Nombre del mapa de puntos',
		value=None,
		key="outpoints"
	)

	st.text_input(
		"Nombre para el archivo de mapa de calor",
		help="Nombre del archivo png que contendrá el mapa de calor.",
		placeholder='Nombre del mapa de calor',
		value=None,
		key="outheat"
	)

	st.selectbox(
		"Posición escala", 
		['upper right', 'upper left', 'lower left'], 
		index=0, 
		key='scale_pos',
		placeholder='Localización de la escala',
		help='Localización de la escala dentro del mapa'
	)

	st.selectbox(
		"Posición inset", 
		['upper right', 'upper left', 'lower left'], 
		index=1, 
		key='inset_pos',
		placeholder='Localización del inset',
		help='Localización del inset dentro del mapa'
	)

	button = st.form_submit_button('Enviar')

	if button:

		if uploaded:
			st.session_state.data = pd.read_csv(uploaded)
			generate_maps()

		else:
			error_window("Es necesario cargar una tabla con registros.")

	if len(st.session_state.error_message) > 0:
		error_window(st.session_state.error_message)
		st.session_state.error_message = ""
		

if st.session_state.dots_ready:

	down_space = st.empty()

	down_space.download_button(
		label="Descargue el mapa de puntos como PNG",
		data=st.session_state.bffr0,
		file_name=f"{st.session_state.outpoints}.png",
		mime="image/png"
	)

if st.session_state.heat_ready:

	down_space_bis = st.empty()

	down_space_bis.download_button(
		label="Descargue el mapa de calor como PNG",
		data=st.session_state.bffr1,
		file_name=f"{st.session_state.outheat}.png",
		mime="image/png"
	)

exit(0)