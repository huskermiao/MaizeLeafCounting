# Zookeeper
Note: this has been deprecated, users are encouraged to use the official zooniverse cli project management tool [Panoptes](https://github.com/zooniverse/Panoptes.git)
```
$ python Zookeeper.py
Usage: Zookeeper.py [OPTIONS] COMMAND [ARGS]...

  Zooniverse project data management tools

Options:
  --help  Show this message and exit.

Commands:
  export    Gets export from zooniverse project
  manifest  Generate a manifest in the directory...
  upload    Uploads images from the image directory to...
```

Zookeeper is a command line interface designed to aid in [Zooniverse](https://www.zooniverse.org/about "About Zooniverse") project management. This cli can upload images, generate image manifests and download annotation exports from zooniverse. 

# Dependencies
To use Zookeeper you must have the panoptes-client api installed as well as click. <br/> 
<br/>
``` 
pip install panoptes-client click
```
or
```
conda install panoptes-client click
```

# Actions:

## Upload
```
$ python Zookeeper.py upload --help
Usage: Zookeeper.py upload [OPTIONS] IMGDIR PROJID

  Uploads images from the image directory to zooniverse project

Options:
  -s, --subject TEXT  Designate subject set id
  -q, --quiet TEXT    Silences output when uploading images to zooniverse
  --help              Show this message and exit.
```

Upload allows users to upload images from the command line.

The following command will upload images from path `/data/images` to a project with id 1234 and subject set id 2345. Zooniverse project ids are listed in the project editting page at the top left corner.

`python -m schnablelab.Zooniverse.Zookeeper upload /data/images 1234 --subject 2345`

## Export

```
$ python ~/schnablelab/Zooniverse/Zookeeper.py export --help              │
Usage: Zookeeper.py export [OPTIONS] PROJID OUTFILE                                                                   │
                                                                                                                      │
  Gets export from zooniverse project                                                                                 │
                                                                                                                      │
Options:                                                                                                              │
  -t, --exp_type TEXT  Specify the type of export. Check Zooniverse help for                                          │
                       available types                                                                                │
  -g, --no_generate    Generate a new export, versus download an existing one                                         │
  --help               Show this message and exit.
```

Export allows project creators to obtain already generated exports from the zooniverse project. To create exports, users must navigate to the 'Data Exports' tab and select 'request new classification export'. After the classification export has been generated, it can be access through the command line with this export function.

![Zooniverse Export image](https://github.com/freemao/pics/blob/master/zookeeper.png)

## Manifest
```
$ python Zookeeper.py manifest --help
Usage: Zookeeper.py manifest [OPTIONS] IMGDIR

  Generate a manifest in the directory specified by imgdir

Options:
  --help  Show this message and exit.
```
Manifests are where you can specify the images that will be uploaded. The upload function uploads only images from the manifest file. This function generates a manifest of all image files inside a specified directory for instances in which all images from the directory are needing to be uploaded. If a user wants to upload selected images (those images can be in different directories), they simply need to generate a csv file named 'manifest.csv' with a header: 'id, filename'. The files will need to be listed with a unique id to avoid filename collisions and the local file path to the image.
