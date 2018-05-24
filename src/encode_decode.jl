include("lpc.jl")

infile = "in.wav"
if length(ARGS) >= 1
    infile = ARGS[1]
end

pcm, fs = wavread(infile)
xs = pcm[:, 1]

out, n, frameTypes = lpc_encode_decode(xs, fs)

outfile = splitext(infile)[1] * "_lp.wav"
dump_voicing_labels(out, fs, n, frameTypes, outfile)


outfile = splitext(infile)[1] * "_labelled.wav"
dump_voicing_labels(xs, fs, n, frameTypes, outfile)
