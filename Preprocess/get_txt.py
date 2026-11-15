import csv  

def csv_to_txt(csv_file, txt_file, delimiter_in=',', delimiter_out=' '):  
    with open(csv_file, mode='r', newline='', encoding='utf-8') as infile, open(txt_file, mode='w', encoding='utf-8') as outfile:  
        csv_reader = csv.reader(infile, delimiter=delimiter_in)  
          
        next(csv_reader)  
          
        for row in csv_reader:  
            outfile.write(delimiter_out.join(row) + '\n')  

csv_to_txt('new_Osaka_acc.csv', 'Osaka_landmark.txt')  
  
print("Data successfully converted from new_data.csv to sh_landmark.txt, with headers removed and commas replaced by spaces.")
