# coding: utf-8

#########################       Documentación       ############################
#
# El presente script utiliza varias librerías espaciales de Python para producir 
# mapas de registros en la ciudad de Bogotá. Los datos de entrada deben corresponder
# a un archivo csv conforme al formato DarwinCore. Aunque no es necesario que el 
# archivo contenga todos los campos DarwinCore ara producir un mapa, si son 
# necesarias las siguientes columnas:
#
# 1. decimalLatitude: Latitud en formato decimal.
#
# 2. decimalLongitude: Longitud en formato decimal.
#
# 3. scientificName: Nombre de la especie.
#
# La ubicación del archivo en el disco duro se debe especificar a través de la 
# variable `CSV_FILE`. El archivo debe estar codificado con el estándar UTF-8,
# tener las columnas separadas por comas y los decimales indicados con puntos.
# 
# A parte de los registros de la especie, es necesario especificar la ubicación 
# de un shapefile que corresponda a los límites de la ciudad de Bogotá o de sus 
# localidades a través de la variable `BOG_SHAPE`.
# 
# Salidas
# 
# La primera salida del script es un mapa de puntos, donde se ubican todos los 
# registros del archivo csv que entran en el polígono de Bogotá (se descarta la
# información del nombre científico). La segunda salida corresponde a un mapa 
# de calor de especies por celda de 2 x 2 km. Ambos mapas son complementados por 
# una escala de 5 km en la esquina superior izquierda y una flecha de norte en 
# la esquina inferior derecha. Adicionalmente, lo archivos de salida son imágenes 
# PNG de 300 dpi, de 12 x 8 pulgadas.
#
# Una vez se cambian las variables necesarias, el script se ejecuta a través de
# la terminal del sistema con el siguiente comando:
#
# python mapper.py
#
################################################################################


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

BOG_SHAPE = "../Datos/GIS/localidades_bogota/Loca.shp"
#CSV_FILE = "../Investigaciones/Ericaceae_Flora_Bogota/Datos/Ericaceae_2025-08-14.csv"
CSV_FILE = "chapi_ericads.csv"
POINTS_FILE = 'map_test.png'
HEAT_MAP_VAR = 'scientificName'
HEAT_MAP_FILE = 'mapa_calor.png'
INSET_POS = 'upper left'
SCALE_POS = 'upper right'
TITLE = 'Ericaceae de Bogotá'

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




dist5km = Geodesic.WGS84.Direct(4.11, -74.11, 0, 5000)
onekm = (((dist5km['lat1'] - dist5km['lat2'])**2 + (dist5km['lon1'] - dist5km['lon2'])**2)**0.5) / 1


data = pd.read_csv(CSV_FILE)
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
	
	plot_ext_lat, plot_ext_lon = get_plot_extent(fpoints, INSET_POS)

######################    Plot first map    #####################################

fig, ax = plt.subplots(figsize=(12, 8))

if plot_ext_lat and plot_ext_lon:
	# Set map limits
	ax.set_xlim(plot_ext_lon)
	ax.set_ylim(plot_ext_lat)

	iax = inset_map(ax, location=INSET_POS, size=2, xticks=[], yticks=[])
	frame.plot(ax=iax, linewidth=0.5)
	indicate_extent(pax=iax, bax=ax, pcrs=4326, bcrs=4326)


frame.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)
fpoints.plot(ax=ax, markersize=2)

insert_scale(ax, dist5km['lat1'], dist5km['lon1'], dist5km['lat2'], 
	dist5km['lon2'], SCALE_POS)

insert_arrow(ax)

plt.title(TITLE, pad=25)
plt.xlabel('Longitud', labelpad=15)
plt.ylabel('Latitud', labelpad=15)
plt.savefig(POINTS_FILE, dpi=300, bbox_inches='tight')


if plot_ext_lat is None and plot_ext_lon is None:

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
	dissolve = merged.loc[merged[HEAT_MAP_VAR].notna()
		].dissolve(
		by="index_cells",
		aggfunc={
			HEAT_MAP_VAR: (lambda i: np.unique(i))   
		}
	)
	dissolve['n_cats'] = dissolve[HEAT_MAP_VAR].apply(lambda m: m.shape[0])
	cell.loc[dissolve.index, 'n_cats'] = dissolve.n_cats.values
	thmax = cell.n_cats.max()


	###################      Plot second map     ###################################

	fig, ax = plt.subplots(figsize=(12, 8))

	if plot_ext_lat and plot_ext_lon:
		# Set map limits
		ax.set_xlim(plot_ext_lon)
		ax.set_ylim(plot_ext_lat)

		iax = inset_map(ax, location=INSET_POS, size=2, xticks=[], yticks=[])
		frame.plot(ax=iax, linewidth=0.5)
		indicate_extent(pax=iax, bax=ax, pcrs=4326, bcrs=4326)
		
	frame.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)
	fpoints.plot(ax=ax, markersize=2)
	f0 = frame.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)
	f1 = cell.plot(ax=ax, column='n_cats', cmap='viridis', vmax=thmax, edgecolor="white", legend=True)

	insert_scale(ax, dist5km['lat1'], dist5km['lon1'], dist5km['lat2'], 
		dist5km['lon2'], SCALE_POS)

	insert_arrow(ax)


	plt.title(TITLE, pad=25)
	plt.xlabel('Longitud', labelpad=15)
	plt.ylabel('Latitud', labelpad=15)
	plt.savefig(HEAT_MAP_FILE, dpi=300, bbox_inches='tight')

exit(0)