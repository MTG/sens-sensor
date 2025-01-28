


---
<!-- PROJECT LOGO -->

<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="data/images/SENS_banner.png" alt="Logo">
  </a>


  <p align="center">

  The Smart Environmental Noise System (SENS) is an advanced sensor technology designed for real-time acoustic monitoring, with a focus on urban environments. Built on a Raspberry Pi platform, SENS captures sound continuously and processes it locally using custom-developed software based on small and efficient artificial intelligence algorithms. SENS calculates acoustic parameters, including Sound Pressure Level (SPL), and makes predictions of the perceptual sound attributes of pleasantness and eventfulness (ISO 12913), along with detecting the presence of specific sound sources such as vehicles, birds, and human activity, among others. To safeguard privacy, all processing occurs directly on the device in real-time ensuring that no audio recordings are permanently stored or transferred. Additionally, the system transmits the extracted audio representation through the wireless network to a remote server, made possible using mobile data connectivity. SENS technology represents an innovative step in environmental noise monitoring, offering real-time processing and robust privacy protection. A single SENS device or a network of them could serve as a powerful tool for understanding the acoustic characteristics of soundscapes with efficiency and flexibility.

  </p>
  
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

This work was supported by the project ''Soundlights: Distributed Open Sensors Network and Citizen Science for the Collective Management of the City's Sound Environments'' (9382417), a collaboration between the <i>Music Technology Group (Universitat Pompeu Fabra)</i> and <i>Bitlab Cooperativa Cultural</i>.

It is funded by <i>BIT Habitat (Ajuntament de Barcelona)</i> under the program <i>La Ciutat Proactiva</i>; and by the <i>IA y Música: Cátedra en Inteligencia Artificial y Música</i> (TSI-100929-2023-1) by the <i>Secretaría de Estado de Digitalización e Inteligencia Artificial</i> and <i>NextGenerationEU</i> under the program <i>Cátedras ENIA 2022</i>. 

<p>
  <img src="data/images/Soundlights_logo.png" alt="Logo" height="70" style="float: left;margin-right: 10px;">
  <img src="data/images/MTG_logo.jpeg" alt="Logo" height="60" style="float: left; margin-right: 10px;">
  <img src="data/images/Bitlab_logo.png" alt="Logo" height="60" style="float: left; margin-right: 10px;">
  <img src="data/images/Logo-Bithabitat.png" alt="Logo" height="60" style="float: left;">
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<p></p>

## Reference
- Amaia Sagasti, Martín Rocamora, Frederic Font: *Prediction of Pleasantness and Eventfulness Perceptual Sound Qualities in Urban Soundscapes* - DCASE Workshop 2024 <a href="https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Sagasti_12.pdf">Paper link DCASE webpage</a>
- Amaia Sagasti Martínez - MASTER THESIS: *Prediction of Pleasantness and Eventfulness Perceptual Sound Qualities in Urban Soundscapes* - Sound and Music Computing Master (Music Technology Group, Universitat Pompeu Fabra - Barcelona) <a href="https://zenodo.org/records/13861445">Master Thesis Report link Zenodo</a>


<!-- GETTING STARTED -->
## Environment set up
This section provides all the necessary information to set up the working environment. 

NOTE: This project is only compatible with 64-bit RaspberryPi architecture. Check your architecture by opening a terminal and running: 
```
uname -m
```

If the output is *aarch64*, you have a 64-bit ARM architecture --> COMPATIBLE

The followiing list details the set up process:
### 1) Download and install  ```Python=3.10.14``` 
Python 3.10.14 <a href=https://www.python.org/downloads/release/python-31014/>download web</a>. 

Follow instructions below (or <a href="https://www.enablegeek.com/tutorial/install-python-on-a-raspberry-pi-step-by-step-guide/#:~:text=Downloading%20Python,-The%20next%20step&text=Here's%20how%20to%20do%20it,latest%20version%20is%20Python%203.10">link</a>)
```
sudo apt-get update

sudo apt-get install build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev

wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tar.xz # or download directly from link above

# Navigate to directory where download is
tar -xvf Python-3.10.14.tar.xz

cd Python-3.10.14

./configure --enable-optimizations # be patient

make -j 4 # be waaay more patient here

sudo make altinstall

python3.10 --version # to verify installation
```


### 2) Install dependencies

It is recommended to create a virtual environment. Example with venv:
```
# Go to home directory

/usr/local/bin/python3.10 -m venv my_env

# to activate
source my_env/bin/activate

# to deactivate
my_env deactivate
```

This code uses <a href="https://github.com/LAION-AI/CLAP">LAION-AI's CLAP model</a>. Install CLAP with:

```
git clone https://github.com/LAION-AI/CLAP.git
```

Finally, install sens-sensor specific requirements. For that, navigate your terminal to the SENS project folder and run:
```
cd sens-sensor

pip install -r requirements.txt
```
Now you are ready to start using sens-sensor repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Raspberry Pi device information
This project is implemented in a RaspberryPi model B. The device has a microphone connected as well as a Mobile Network Module with a SIM card. Additionally, three LED pins are connected.

## Run code

The following three images indicate the 3 main processes that create SENS working. Each process is called through a different python script that is called in different terminal windows.

<img src="data/images/main_capture.png" alt="audio capture process" width="75%">
<img src="data/images/main_process.png" alt="audio processing process" width="75%">
<img src="data/images/main_send.png" alt="result sending process" width="75%">

<i>NOTE: All inputs indicated in orange color indicate that they are defined in the ***parameters.py*** file.</i>

#### Other scripts


<!-- LICENSE -->
## License
See `LICENSE` for more information.<p align="right">(<a href="#readme-top">back to top</a>)</p>-->
