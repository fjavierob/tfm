from matplotlib import pyplot as plt
import code

SAVE_DIR = 'resultados/graficas/'

# NOMBRE   = 'knn_enventanado_valencia.png'
# TITULO   = "ANÁLISIS DE ENVENTANADO PARA CLASIFICACIÓN DE VALENCIA"
# texto    = ['61.1%', '63.2%', '66.3%', '67.8%', '66.7%', '66.7%', '69.1%', '67.4%', '68.2%']
# accs     = [61.1, 63.2, 66.3, 67.8, 66.7, 66.7, 69.1, 67.4, 68.2]

NOMBRE   = 'muestras_significativas_vs_neutras.png'
TITULO   = "MUESTRAS SIGNIFICATIVAS Y NEUTRAS"
texto    = ['54.98%', '76.49%', '58.29%', '76.45%']
accs     = [54.98, 76.49, 58.29, 76.45]



XTICKS_LABELS = ['VALENCIA', 'EXCITACIÓN']
labels = ['Muestras neutras', 'Muestras significativas']
colors = ['blue', 'green']


w   = 0.6  # the width of the bars
sep = 1.5*w
nmults   = 2
nkernels = 2
largo = (w*nkernels+sep)*nmults+1
fig, ax = plt.subplots(figsize=(largo,5))
tickslabels = XTICKS_LABELS

# Posicion xticks
offset=w/2
pos = []
for x in range(nmults):
    posx = offset + nkernels*w/2+x*sep
    pos.append(posx)
    offset = offset + nkernels*w
k = 0
i = 0
label    = True
for a in range(1,nmults+1):
    for c in range(1,nkernels+1):
        if label:
            rect = ax.bar((k+1)*w, accs[i], w, label=labels[c-1], edgecolor='black', color=colors[c-1])
        else:
            rect = ax.bar((k+1)*w, accs[i], w, edgecolor='black', color=colors[c-1])
        x = (rect[0].get_x() + rect[0].get_width()/2.)
        y = 1.03*rect[0].get_height()
        ax.text(x, y, texto[i], ha='center', va='bottom', fontsize=10)
        k+=1
        i+=1
    label = False
    k+=sep/w
    plt.gca().set_prop_cycle(None)
ax.legend()
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True ,      # ticks along the bottom edge are on
    top=False,         # ticks along the top edge are off
    labelbottom=True,
    rotation=10) # labels along the bottom edge are off

ax.set_xticks(pos)
ax.set_xticklabels(tickslabels)
ax.set(ylabel='Tasa acierto [%]')
ax.set_ylim(ymax=100)
fig.suptitle(TITULO)
fig.savefig(SAVE_DIR+NOMBRE)