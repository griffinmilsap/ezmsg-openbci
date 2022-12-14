import ezmsg.core as ez

from ezmsg.openbci import OpenBCISource, OpenBCISourceSettings

from ezmsg.testing.debuglog import DebugLog


class OpenBCITestSystem(ez.Collection):

    SETTINGS: OpenBCISourceSettings

    SOURCE = OpenBCISource()
    DEBUG = DebugLog()

    def configure(self) -> None:
        self.SOURCE.apply_settings(self.SETTINGS)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.SOURCE.OUTPUT_SIGNAL, self.DEBUG.INPUT),
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ezmsg.openbci test script'
    )

    parser.add_argument(
        '--device',
        type=str,
        help='serial port to pull data from',
        default='simulator'
    )

    parser.add_argument(
        '--blocksize',
        type=int,
        help='sample block size @ 500 Hz',
        default=50
    )

    parser.add_argument(
        '--poll',
        type=float,
        help='poll Rate (Hz). 0 for auto-config',
        default=0.0
    )

    class Args:
        device: str
        blocksize: int
        poll: float

    args = parser.parse_args(namespace=Args)

    settings = OpenBCISourceSettings(
        device=args.device,
        blocksize=args.blocksize,
        poll_rate=None if args.poll <= 0 else args.poll
    )

    system = OpenBCITestSystem(settings)
    ez.run(system)