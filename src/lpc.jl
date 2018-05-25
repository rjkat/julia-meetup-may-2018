using WAV
using PlotlyJS
using DSP
using DSP.Filters

function center_clip(xs; thresh=0.3)
    cl = thresh * maximum(abs.(xs))
    [abs(x) > cl ? (x - (sign(x) * cl)) : 0.0 for x in xs]
end

function clip(xs)
    max.(min.(xs, 1.), -1.)
end

function freq_trace(ys, fs; name="spectrum")
    nbins = length(ys)
    xs = round.(Int, fs / 2 * (0:nbins-1) ./ (nbins))
    spec = 20.*log10.(ys)
    scatter(x=xs, y=spec, name=name)
end

function fft_trace(x, fs; name="spectrum")
    ys = abs.(fftshift(fft(x)))
    nbins = div(length(ys), 2)
    xs = round.(Int, fs / 2 * (0:nbins-1) ./ (nbins))
    spec = 20.*log10.(ys[nbins+1:end])
    scatter(x=xs, y=spec, name=name)
end

function lpc_trace(x, fs, lpcOrder; name="LP order $lpcOrder")
    A, erra = lpc(x, lpcOrder)
    G = sqrt(erra)
    ws = linspace(0, π, length(x))
    Aw = zeros(Complex, size(ws))
    for (i, w) in enumerate(ws)
        Aw[i] = 1.
        for k = 1:length(A)
            Aw[i] += A[k] * exp(-im * w * k)
        end
    end
    freq_trace(abs.(G./Aw), fs; name=name)
end

function plot_spectrum(xs, fs)
    if eltype(xs) <: Vector
        traces = [fft_trace(x, fs) for x in xs]
    else
        traces = [fft_trace(xs, fs)]
    end
    plot(traces,
        Layout(
            yaxis_title="magnitude (dB)",
            xaxis_type="log",
            xaxis_title="frequency (Hz)"
        )
    )
end

function plot_lpc_spectrum(xs, fs; lpcOrders=[10,20,30], title="LP spectral envelope")
    plot([fft_trace(xs, fs; name="actual"); [lpc_trace(x, fs, o) for o in lpcOrders]],
        Layout(
            title=title,
            yaxis_title="magnitude (dB)",
            xaxis_type="log",
            xaxis_title="frequency (Hz)"
        )
    )
end

# http://www.emptyloop.com/technotes/a%20tutorial%20on%20linear%20prediction%20and%20levinson-durbin.pdf
# https://gist.github.com/jameslyons/8779402
function levinson{T<:Real}(r::Vector{T}, m::Int)
    A = zeros(m, m)
    err = zeros(1, m)

    # for k = 1
    A[1, 1] = - (r[2] / r[1])
    err[1] = r[1] * (1 - A[1, 1] ^ 2)

    # for k = 2,3,4,..m
    for k = 2:m
        λ = -(sum(A[k-1, 1:k-1] .* r[k:-1:2]) + r[k+1])/err[k-1]
        A[k, k] = λ
        A[k, 1:k-1] .= A[k-1, 1:k-1] + λ * A[k-1, k-1:-1:1]
        err[k] = err[k-1] * (1 - λ^2)
    end
    A[m, :], err[m]
end

function pitch_period_samples(xs; min_period_samples=50)
    findmax(xcorr(xs, xs)[length(xs)+min_period_samples:end])[2] + min_period_samples
end

function impulse_train(n; period=n)
    nrepeats = div(n + period - 1, period)
    repeat([1.; [0. for _ = 1:period-1]], outer=nrepeats)[1:n]
end

function lpc(xs, m)
    r = xcorr(xs, xs)[length(xs):end]
    A, err = levinson(r, m)
    return A, err
end

function zero_crossing_rate(x)
    sum(abs.(diff(0.5 .* sign.(x)))) / length(x)
end

@enum FrameType VoicedFrame UnvoicedFrame

function dump_voicing_labels(x, fs, block_size_samples, labels, outfile)
    markers = Dict{UInt32, WAVMarker}()
    j = 0
    last_label = labels[1]
    i = 1
    for l in labels[2:end]
        if l != last_label
            markers[length(markers)] = WAVMarker(
                last_label == VoicedFrame ? "voiced" : "unvoiced",
                floor(Int, j * block_size_samples),
                floor(Int, (i - j) * block_size_samples)
            )
            j = i
            last_label = l
        end
        i += 1
    end

    l = labels[end]
    markers[length(markers)] = WAVMarker(
        last_label == VoicedFrame ? "voiced" : "unvoiced",
        floor(Int, j * block_size_samples),
        floor(Int, (length(labels) - j) * block_size_samples)
    )

    out_chunks = wav_cue_write(markers)
    wavwrite(x, outfile, Fs=fs, compression=WAV.WAVE_FORMAT_PCM, chunks=out_chunks)
end

function lpc_encode_decode(xs, fs; excitationFile=nothing, lpcOrder=30, blockSizeMs=15, unvoicedZCR=0.12, unvoicedHz=300.)
    out = zeros(xs)

    if excitationFile != nothing
        excitationSamples, excitationFs = wavread(excitationFile)
        assert(excitationFs == fs)
    end
    n = ceil(Int, blockSizeMs / 1000. * fs)
    noverlap = div(n, 2)
    nblocks = div(length(xs), n)
    frameTypes = []

    index = 1
    lastOutput = zeros(n)
    while index + n < length(xs)
        block = xs[index:(index+n-1)]
        block = hamming(n) .* block
        z = zero_crossing_rate(block)
        A, erra = lpc(block, lpcOrder)
        G = sqrt(erra)
        p = pitch_period_samples(center_clip(block))
        freq_hz = fs / p
        ft = (z > unvoicedZCR || freq_hz > unvoicedHz) ? UnvoicedFrame : VoicedFrame
        
        if excitationFile == nothing
            if ft == VoicedFrame
                excitation = impulse_train(n; period=(p*4))
            else
                excitation = (rand(n) - 0.5) * 0.1
            end
        else
            excitation = excitationSamples[index:(index+n-1)]
        end
        push!(frameTypes, ft)
        f = PolynomialRatio([G], [1; A])
        predicted = hamming(n) .* filt(f, excitation)
        out[index:(index+noverlap-1)] = lastOutput[(noverlap+1):n]
        out[index:(index+n-1)] .+= predicted
        lastOutput = predicted
        index += noverlap
    end

    return out, noverlap, frameTypes
end
