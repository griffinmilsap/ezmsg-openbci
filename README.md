# ezmsg.openbci

OpenBCI Cyton serial interface for `ezmsg`

## Installation
`pip install ezmsg-openbci@git+https://github.com/griffinmilsap/ezmsg-openbci`

Or, just grab `ezmsg/openbci/__init__.py` and drop it in your project; it is only one file.

## Dependencies
* `ezmsg-sigproc`
* `pyserial>=3.5`

## Setup (Development)
1. Install `ezmsg` either using `pip install ezmsg` or set up the repo for development as described in the `ezmsg` readme.
2. `cd` to this directory (`ezmsg-openbci`) and run `pip install -e .`
3. Signal processing modules are available under `import ezmsg.openbci`

## Hardware Setup
* __This module does not work with the vanilla OpenBCI Cyton firmware.__
* __This module requires hard-wire serial connection to a serial port on the machine this module is running on.__
* __This module will not function with the wireless USB dongle that comes with the standard OpenBCI Cyton__

To use this, you will need to compile/flash the firmware in the `firmware` folder to the OpenBCI Cyton dev board.  This has become progressively more difficult as the software landscape has evolved since this product launched a _decade_ ago.  You can find a tutorial on flashing board firmware here: [https://docs.openbci.com/Cyton/CytonProgram/]

This custom firmware uses a few specially configured pins to output data directly to the host machine over serial.  When compiling your firmware, you will need to edit a `BoardDefs.h` file as specified here: [https://github.com/OpenBCI/OpenBCI_Cyton_Library#begindebug].  Contrary to what the documentation says at this link, this custom firmware will use 230400 Baud communications over serial.

After re-flashing the custom firmware to the Cyton, you will use Pin D11 on J4 (TX) and Pin D12 on J3 (RX) and a common ground (I've used GND pin on J3) to connect to a 3.3v serial bus on the receiving machine.  I use a Raspberry Pi Zero / (2W) as my host board, and I connect these directly to pins on my Raspberry Pi as follows: 
* D11 (Cyton TX) -> IO15 (Pi RXD)
* D12 (Cyton RX) -> IO14 (Pi TXD)
* GND (Cyton) -> GND Pin across from IO15/RXD (Pi GND)

Enabling 230400 baud serial communications on the Pi Zero requires some custom configuration. [Coming soon]






