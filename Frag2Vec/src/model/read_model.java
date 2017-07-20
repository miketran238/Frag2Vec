package model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.Scanner;

public class read_model {
	
	public static Map<String,Boolean> oracle_map = new HashMap<>();
	public static Map<String,Map<String,Double>> result_map = new HashMap<>();
	public final static int num_relevant_pair 	= 56; 
//	public final static String model_name = "col_combine_max_range";
	public final static String model_name = "col_concatinate_max_range";
//	public final static String model_name = "combine2_max_test_norange";
	public static void main(String args[]) throws IOException{
		
		/*		Initialize Log file
		Store info of training */

		File file_output	= new File ("resources/output/result/"+model_name+"_result");
		FileWriter fWriter	= new FileWriter (file_output);
		PrintWriter writer	= new PrintWriter (fWriter);
		
		/*
		 * Step 1: Read Model file and Build API vector 
		 */
		
		File file = new File ("resources/output/model/"+model_name);
		
		Scanner sc = new Scanner(file);
		while (sc.hasNextLine()){
			Map<String,Double> value = new HashMap<>();
			String key = sc.nextLine();
			String result = sc.nextLine();
			Scanner sc_in = new Scanner (result);
			while (sc_in.hasNext()){
				String DOC_ID		= sc_in.next();
				double cosine_simi 	= sc_in.nextDouble();
				value.put(DOC_ID, cosine_simi);
			}
			value = sortByValue(value);
			result_map.put(key, value);
			sc_in.close();
		}
		sc.close();
		
		for (Entry<String, Map<String, Double>> entry : result_map.entrySet()){
			System.out.println(entry.getKey());
			System.out.println(entry.getValue().toString());
		}
		// *****************************************************************************
		
		/*
		 * Step 2:	Read Oracle file and build a Map
		 * Key		= 0_Collections -> relation btw DOC_0 and class Collections 
		 * Value	= False (not relevant) or True (relevant)
		*/
		File file_oracle = new File ("resources/input/tutorial_col_oracle");
		Scanner sc_oracle = new Scanner (file_oracle);
		int i = 0;
		while(sc_oracle.hasNextLine()){
			Scanner line = new Scanner(sc_oracle.nextLine());
			while (line.hasNext()){
				String token 	= line.next();
				String key		= i + "_" + token;
				boolean value	= (line.nextInt() == 1) ? true : false;
				oracle_map.put(key, value);
			}				
			i++;
			line.close();
		}
		sc_oracle.close();
		// *****************************************************************************
		
		/*
		 * Step 3: Precision & Recall with Threshold
		 */
		
		for (int j = 0; j <= 100; j++){
//		for (double T = 0; T <= 1; T = T + 0.01){
			int TP = 0;
			int FP = 0;
			int num_retrieve = 0;
			double T = j * 0.01;
			for (Entry<String,Map<String,Double>> entry : result_map.entrySet()){
				String class_name = entry.getKey();
				for (Entry<String,Double> entry_in : entry.getValue().entrySet()){
					if (entry_in.getValue() >= T){
						String key = entry_in.getKey().replaceAll("[^0-9]", "");
						key = key + "_" + class_name;
						if (oracle_map.containsKey(key)){
							if (oracle_map.get(key))
								TP ++;
							else
								FP ++;
						}else{
							FP ++;
						}
						num_retrieve ++;
					}
				}
			}
			double precision = 100 * (double)TP / num_retrieve;
			double recall    = 100 * (double)TP / num_relevant_pair;
			System.out.println(T+" "+precision+" "+recall + " " + TP);
			writer.println(T+" "+precision+" "+recall);
		}
		// *****************************************************************************
//		System.out.println("TP: "+TP);
		writer.close();

	}
	
	// Sort Map by Value
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
		return map.entrySet()
				.stream()
				.sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
				.collect(Collectors.toMap(
						Map.Entry::getKey, 
						Map.Entry::getValue, 
						(e1, e2) -> e1, 
						LinkedHashMap::new
						));
	}
}
