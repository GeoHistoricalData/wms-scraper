wms-scraper
==============
WMS-scraper is a command line tool for downloading parts of a georeferenced map available through a Web Map Service.

**Example usage**
```shell
scrapwms --wms https://ows.terrestris.de/osm-gray/service --crs EPSG:3857 --bbox -10525023,4744656,-10516206,4745841  OSM-WMS output.tif
```


GDAL
-------
WMS-scraper relies on the Python wrapper of GDAL to download a region as a mosaic of smaller images that are eventually merged into a single GeoTiff.

Before installing `wms-scraper` you will need the GDAL development libraries.
```shell
apt-get install libgdal-dev
```


Install
-------
From PyPiTest:
```shell
pip install -i https://test.pypi.org/simple/ wms-scraper
```

From sources, in development mode:
```bash
pip install -e .
```


**Troubleshooting GDAL**

The version of the `gdal` Python wrapper resolved by pip may not be compatible with the native GDAL library installed on your system, causing the installation to fail.
In such case you will need to downgrade `gdal` to match the version of the native library.

Get the version of GDAL currently installed on your system and force reinstalling `gdal`:
```shell
gdalinfo --version
# e.g. "GDAL 3.4.1, released 2021/12/27". Here the version is "3.4.1".

# Replace GDAL_VERSION with the version from gdalinfo.
pip install --force-reinstall -v "gdal==GDAL_VERSION"
```


Usage
-----

```
$ scrapwms --help

Usage: scrapwms [OPTIONS] LAYER OUTPUT

  LAYER: Name of the WM(T)S layer to scrap.

  OUTPUT: Path to the GeoTiff file to create. Note that the outputed file will be of the form `{output}.{clipping}[_{region}].tif` where:
      - `clipping` is the name of the method used to define the regions,
      either "grid", "bbox" or "sheetfile";
      - `region` is the index of the corresponding region when clipping
      multiple regions with a grid or a sheetfile.

Options:
  --wms TEXT                  WMS endpoint URL. For instance
                              `https://wms.openstreetmap.fr/wms`.
  --grid FILENAME             A GEOJSON grid of regions with bboxes as
                              properties `{'left': xmin, 'bottom': ymin,
                              'right': xmax, 'top': ymax}`.  NOTE: this
                              argument is mutually exclusive with --bbox and
                              --sheetfile.
  --bbox TEXT                 A bounding box formatted as
                              `xmin,ymin,xmax,ymax`.  NOTE: this argument is
                              mutually exclusive with --grid and --sheetfile.
  --sheetfile FILENAME        A GEOJSON file containing bounding boxes
                              geometries. See also --sheetnumber and
                              --sheetnumbername. NOTE: this argument is
                              mutually exclusive with --grid and --bbox.
  -n, --sheetnumber INTEGER   Filter a sheetfile to extract only the sheet
                              with this number.
  -a, --sheetnumbername TEXT  The property on which --sheetfile will filter.
  -r, --resolution FLOAT      Resolution of the extracted images, in pixels
                              per map unit. Defaults to 1.0px/mu. Map units
                              depends on the CRS (see --crs).
  --wmts                      Use WMTS.
  --crs TEXT                  Coordinate reference system, e.g. EPSG:3857.
  -v, --verbose               Verbose mode
  -q, --quiet                 Quiet mode
  --help                      Show this message and exit.
```
