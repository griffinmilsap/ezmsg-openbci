import asyncio
import time

from dataclasses import dataclass, field
from enum import Enum

import serial

import numpy as np
import numpy.typing as npt

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.window import WindowSettings, Window

from typing import (
    Optional,
    Tuple,
    AsyncGenerator,
    List
)

@dataclass
class OpenBCIEEGMessage(AxisArray):
    ch_names: List[Optional[str]] = field(default_factory=list)


class PowerStatus(Enum):
    POWER_ON = 0  # Default
    POWER_OFF = 1


class GainState(Enum):
    GAIN_1 = (0, 1.0)
    GAIN_2 = (1, 2.0)
    GAIN_4 = (2, 4.0)
    GAIN_6 = (3, 6.0)
    GAIN_8 = (4, 8.0)
    GAIN_12 = (5, 12.0)
    GAIN_24 = (6, 24.0)  # Default

    def __init__(self, code, gain):
        self._code = code
        self._gain = gain

    @property
    def code(self) -> int:
        return self._code

    @property
    def gain(self) -> float:
        return self._gain


class InputSource(Enum):
    NORMAL = 0  # Default
    SHORTED = 1
    BIAS_MEAS = 2
    MVDD = 3
    TEMP = 4
    TESTSIG = 5
    BIAS_DRP = 6
    BIAS_DRN = 7


class BiasSetting(Enum):
    REMOVE = 0
    INCLUDE = 1  # Default


class SRB2Setting(Enum):
    DISCONNECT = 0
    CONNECT_TO_P = 1  # Default


class SRB1Setting(Enum):
    DISCONNECT = 0  # Default
    CONNECT_TO_N = 1


class OpenBCIChannelSetting(ez.Settings):

    name: Optional[str] = None
    power: PowerStatus = PowerStatus.POWER_ON
    gain: GainState = GainState.GAIN_24
    input: InputSource = InputSource.NORMAL
    bias: BiasSetting = BiasSetting.INCLUDE
    srb2: SRB2Setting = SRB2Setting.CONNECT_TO_P
    srb1: SRB1Setting = SRB1Setting.DISCONNECT


class OpenBCIChannelConfigSettings(ez.Settings):

    ch_setting: Tuple[OpenBCIChannelSetting, ...] = field(
        default_factory=lambda: tuple([
            OpenBCIChannelSetting()
            for _ in range(8)
        ])
    )

    def channel_indices_by_power_status(self,
                                        status: PowerStatus = PowerStatus.POWER_ON
                                        ) -> Tuple[int]:

        channel_indices = [
            idx for idx, setting in enumerate(self.ch_setting)
            if setting.power == status
        ]

        return tuple(channel_indices)


@dataclass
class OpenBCISerialStatus:
    connected: bool


class OpenBCISerialSettings(ez.Settings):
    device: str
    blocksize: int
    ch_config: OpenBCIChannelConfigSettings
    poll_rate: Optional[float] = 50  # Hz

    # Constants for this incarnation of OpenBCI
    start_sep: bytes = field(default=b'C', init=False)
    msg_size: int = field(default=36, init=False)
    baudrate: int = field(default=230400, init=False)
    sampling_rate: float = field(default=500.0, init=False)
    vref: float = field(default=4.5, init=False)  # Volts, for calibration


class OpenBCISerialState(ez.State):
    count: Optional[int] = None
    incoming: "asyncio.Queue[ bytes ]" = field(
        default_factory=asyncio.Queue
    )
    outgoing: "asyncio.Queue[ bytes ]" = field(
        default_factory=asyncio.Queue
    )


class OpenBCISerial(ez.Unit):

    SETTINGS: OpenBCISerialSettings
    STATE: OpenBCISerialState

    INPUT_BYTES = ez.InputStream(bytes)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_STATUS = ez.OutputStream(OpenBCISerialStatus)

    @ez.publisher(OUTPUT_STATUS)
    async def handle_port(self) -> AsyncGenerator:

        port = None
        if self.SETTINGS.device != 'simulator':
            port = serial.Serial(
                self.SETTINGS.device,
                baudrate=self.SETTINGS.baudrate,
                timeout=0,
            )

        buf = bytearray()
        chunk_size = self.SETTINGS.msg_size * self.SETTINGS.blocksize
        chunk_period = (1.0 / self.SETTINGS.sampling_rate) * self.SETTINGS.blocksize
        poll_period = chunk_period / 4
        if self.SETTINGS.poll_rate is not None:
            poll_period = (1.0 / self.SETTINGS.poll_rate)

        yield (self.OUTPUT_STATUS,
               OpenBCISerialStatus(
                   connected=True
               )
               )

        if port is not None:
            while True:
                await asyncio.sleep(poll_period)
                if not self.STATE.outgoing.empty():
                    port.write(self.STATE.outgoing.get_nowait())
                    port.flush()
                data = port.read(chunk_size)
                buf.extend(data)

                while len(buf) > chunk_size:
                    idx = buf.find(self.SETTINGS.start_sep)

                    # Message start not found in whole buffer.. wow
                    if idx < 0:
                        buf.clear()
                        break

                    # We might not have enough data yet..
                    if len(buf) < (idx + chunk_size):
                        break

                    # Check for frame alignment
                    if buf[idx + chunk_size - self.SETTINGS.msg_size] == buf[idx]:
                        msg = bytes(buf[idx: (idx + chunk_size)])
                        self.STATE.incoming.put_nowait(msg)
                        buf = buf[(idx + chunk_size):]
                    else:
                        # We don't have alignment.. resynchronize
                        buf = buf[(idx + 1):]
        else:
            n_ch = len(self.SETTINGS.ch_config.ch_setting)
            msg_counter = 0
            start_byte_vec = np.array(bytearray(b'C' * self.SETTINGS.blocksize))[:, np.newaxis]
            zero_vec = np.zeros((self.SETTINGS.blocksize, 1), dtype=np.uint8)
            while True:
                await asyncio.sleep(chunk_period)
                block = np.random.uniform(
                    low=-2**23, high=2**23,
                    size=(self.SETTINGS.blocksize, n_ch)
                ).astype(np.int32)

                byte_arr = bytearray(block.tobytes())
                byte_arr = np.array(byte_arr).reshape(self.SETTINGS.blocksize, n_ch * 4)
                count = np.arange(msg_counter, msg_counter + self.SETTINGS.blocksize) % 256
                count = count.astype(np.uint8)[:, np.newaxis]
                msg_counter = count[-1]

                byte_arr = np.concatenate((start_byte_vec, count, zero_vec, zero_vec, byte_arr), axis=1)
                self.STATE.incoming.put_nowait(byte_arr.tobytes())

    @ez.subscriber(INPUT_BYTES)
    async def queue_tx(self, message: bytes) -> None:
        self.STATE.outgoing.put_nowait(message)

    @ez.publisher(OUTPUT_SIGNAL)
    async def pub_data(self) -> AsyncGenerator:

        ch_setting = self.SETTINGS.ch_config.ch_setting
        enabled_channels = self.SETTINGS.ch_config.channel_indices_by_power_status()

        ch_names = [
            f'Ch{i+1}'
            if ch_setting[i].name is None else ch_setting[i].name
            for i in enabled_channels
        ]

        gains: np.ndarray = np.array([setting.gain.gain
                                      for setting in self.SETTINGS.ch_config.ch_setting
                                      ])

        ad_to_volts = self.SETTINGS.vref / (2**23 - 1)

        def calibrate(arr): return ((arr * ad_to_volts) / gains).astype(np.float32)
        enabled_channels = self.SETTINGS.ch_config.channel_indices_by_power_status()

        while True:
            data_bytes = await self.STATE.incoming.get()
            data: np.ndarray = np.frombuffer(data_bytes, dtype=np.int32)
            data = data.reshape((
                self.SETTINGS.blocksize,
                # We have 4 bytes of extra information per channel
                len(self.SETTINGS.ch_config.ch_setting) + 1
            ))

            # TODO: Check count for lost packets

            cur_signal = calibrate(data[:, 1:])  # time x channel

            # TODO: remove power-down channels

            cur_signal = cur_signal[:, enabled_channels]

            out_msg = OpenBCIEEGMessage( 
                cur_signal, 
                dims = ['time', 'ch'], 
                axes = dict( 
                    time = AxisArray.Axis.TimeAxis( 
                        fs = self.SETTINGS.sampling_rate, 
                        offset = time.time()
                    ),
                ),
                ch_names = ch_names
            )

            yield (self.OUTPUT_SIGNAL, out_msg)


class OpenBCIControllerSettings(ez.Settings):
    ch_config: OpenBCIChannelConfigSettings
    impedance: bool


class OpenBCIController(ez.Unit):
    SETTINGS: OpenBCIControllerSettings

    INPUT_STATUS = ez.InputStream(OpenBCISerialStatus)
    OUTPUT_BYTES = ez.OutputStream(bytes)

    @ez.subscriber(INPUT_STATUS)
    @ez.publisher(OUTPUT_BYTES)
    async def startup(self, message: OpenBCISerialStatus) -> AsyncGenerator:
        print('Configuring OpenBCI...')
        async for msg in self.configure_channels():
            yield msg
        yield (self.OUTPUT_BYTES,
               f'g{"1" if self.SETTINGS.impedance else "0"}'.encode('ascii')
               )
        print('Startup Complete')
        raise ez.Complete

    async def configure_channels(self) -> AsyncGenerator:
        # https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
        for ch_idx, ch_setting in enumerate(self.SETTINGS.ch_config.ch_setting):
            yield(self.OUTPUT_BYTES,
                  ''.join([
                      'x',
                      str(ch_idx + 1),
                      str(ch_setting.power.value),
                      str(ch_setting.gain.code),
                      str(ch_setting.input.value),
                      str(ch_setting.bias.value),
                      str(ch_setting.srb2.value),
                      str(ch_setting.srb1.value),
                      'X'
                  ]).encode('ascii')
                  )

        await asyncio.sleep(5.0)


@dataclass
class OpenBCIImpedanceMessage:
    impedance: np.ndarray


class OpenBCIImpedanceSettings(ez.Settings):
    enable: bool
    time_axis: Optional[str] = 'time'

    # For this implementation
    drive_current: float = field(default=6.0e-9, init=False)  # Amps
    series_resistor: float = field(default=2200.0, init=False)  # Ohms


class OpenBCIImpedance(ez.Unit):
    SETTINGS: OpenBCIImpedanceSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_IMP = ez.OutputStream(OpenBCIImpedanceMessage)

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_IMP)
    async def on_data(self, message: AxisArray) -> AsyncGenerator:

        axis_name = self.SETTINGS.time_axis
        if axis_name is None:
            axis_name = message.dims[0]
        axis_idx = message.get_axis_idx(axis_name)
        axis = message.get_axis(axis_name)
        fs = 1.0 / axis.gain

        if self.SETTINGS.enable:

            n_time = message.data.shape[axis_idx]
            window = np.hamming(n_time)

            view = np.moveaxis(message.data, axis_idx, -1)
            fft = np.fft.fft((view * window)) / n_time  # ch x freq
            freqs = np.fft.fftfreq(n_time, d=1.0 / fs)
            imp_idx = np.argmin(np.abs(freqs - (fs / 4.0)))  # our impedance tone idx
            imp = (np.abs(2.0 * fft) * 2.0 / self.SETTINGS.drive_current)[:, imp_idx]
            imp -= np.clip(self.SETTINGS.series_resistor, 0, None)

            yield(self.OUTPUT_IMP, OpenBCIImpedanceMessage(impedance=imp))


class OpenBCISourceSettings(ez.Settings):
    device: str
    blocksize: int
    impedance: bool = False
    poll_rate: Optional[float] = None
    ch_config: OpenBCIChannelConfigSettings = field(
        default_factory=OpenBCIChannelConfigSettings
    )


class OpenBCISource(ez.Collection):

    SETTINGS: OpenBCISourceSettings

    # Raw Serial Access
    INPUT_SERIAL = ez.InputStream(bytes)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_STATUS = ez.OutputStream(OpenBCISerialStatus)
    OUTPUT_IMP = ez.OutputStream(OpenBCIImpedanceMessage)

    # Subunits
    SERIAL = OpenBCISerial()
    CONTROLLER = OpenBCIController()
    WINDOW = Window()
    IMPEDANCE = OpenBCIImpedance()

    def configure(self) -> None:
        self.SERIAL.apply_settings(
            OpenBCISerialSettings(
                device=self.SETTINGS.device,
                blocksize=self.SETTINGS.blocksize,
                ch_config=self.SETTINGS.ch_config,
                poll_rate=self.SETTINGS.poll_rate
            )
        )

        self.WINDOW.apply_settings(
            WindowSettings(
                window_dur=1.0,  # seconds
                window_shift=1.0,  # seconds
            )
        )

        self.IMPEDANCE.apply_settings(
            OpenBCIImpedanceSettings(
                enable=self.SETTINGS.impedance
            )
        )

        self.CONTROLLER.apply_settings(
            OpenBCIControllerSettings(
                ch_config=self.SETTINGS.ch_config,
                impedance=self.SETTINGS.impedance
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SERIAL, self.SERIAL.INPUT_BYTES),
            (self.SERIAL.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
            (self.SERIAL.OUTPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL),
            (self.WINDOW.OUTPUT_SIGNAL, self.IMPEDANCE.INPUT_SIGNAL),
            (self.IMPEDANCE.OUTPUT_IMP, self.OUTPUT_IMP),
            (self.SERIAL.OUTPUT_STATUS, self.OUTPUT_STATUS),
            (self.SERIAL.OUTPUT_STATUS, self.CONTROLLER.INPUT_STATUS),
            (self.CONTROLLER.OUTPUT_BYTES, self.SERIAL.INPUT_BYTES),
        )
    
    def process_components(self) -> Tuple[ ez.Component, ... ]:
        return (
            self.SERIAL,
        )