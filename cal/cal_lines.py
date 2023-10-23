import matplotlib.pyplot as plt

lines = ['MgKa','AlKa','SiKa','ClKa','KKa','TiKa','FeKa','CuKa']
energies = [1254,1487,1740,2622,3314,4511,6404,8048]

for line,energy in zip(lines,energies):
    plt.axvline(energy,color='k')
    plt.text(energy+10,.6,line,rotation=90)
plt.xlabel('Energy [eV]')
plt.show()