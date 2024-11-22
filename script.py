import os

filename = 'experiment_results/MACBF.csv'
while True:
    if os.path.exists(filename):
        with open(filename) as file:
            lines = file.readlines()
            print(lines)
            new_file = open("experiment_results/MACBF_first.csv", "w+")
            new_file.writelines(lines)
            new_file.close()

        os.system('cp experiment_results/MACBF.csv experiment_results/MACBF2.csv')
        break
