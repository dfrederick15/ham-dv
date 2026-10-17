"""HamDV - live video and audio over an amateur radio FM voice channel.

The physical layer is a single-carrier, frequency-domain-equalised (SC-FDE)
waveform that fits inside the 300-3000 Hz audio passband of an FM voice
repeater.  Everything above the audio interface is software: LDPC forward
error correction, a low-rate LPC vocoder for speech, and two video
transports (a digital DCT/arithmetic-coded codec and an analog, linearly
mapped "SoftCast" style transport).
"""
__version__ = "0.1.0"
