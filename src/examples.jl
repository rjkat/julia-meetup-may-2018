include("lpc.jl")

pcm, fs = wavread("voiced.wav")
x = pcm[:, 1]
pv = plot_lpc_spectrum(x, fs; title="LP spectral envelope (voiced)")

pcm, fs = wavread("unvoiced.wav")
x = pcm[:, 1]
pu = plot_lpc_spectrum(x, fs; title="LP spectral envelope (unvoiced)")

p = [pv; pu]
PlotlyJS.relayout!(p; width=800)
PlotlyJS.savefig(p, "lp.html"; js=:embed)
