RMSOverlap : UGen {
    *ar { arg in=0.0, winSize=1024, hopSize=512, winType=1, zeroPad=0.0, winAlign=1, normType=0, fixedNorm=1.0;
        ^this.multiNew('audio', in, winSize, hopSize, winType, zeroPad, winAlign, normType, fixedNorm)
    }

    // No .kr method provided as RMS is typically processed at audio rate.
}
