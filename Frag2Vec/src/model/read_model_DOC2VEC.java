package model;

import java.io.File;
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

public class read_model_DOC2VEC {

	public static Map<String,Boolean> oracle_map = new HashMap<>();
	public static Map<String,Boolean> FRAPT_map = new HashMap<>();
	public static Map<String,Map<String,Double>> result_map = new HashMap<>();

	public static Map<String,Map<String,Double>> DOC2VEC_map = new HashMap<>();


	public final static int num_relevant_pair 	= 42; 
	public final static int k_top = 10;
	public final static String model_name = "col_concatinate_DOC2VEC_FULL";

	public static void main(String args[]) throws IOException
	{
		readDoc2VecModel();
		readOracleResult();
		readFrapt();
		resultDoc2Vec();
		resultFraptIn();
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
	
	public static void readDoc2VecModel() throws IOException
	{
		/*
		 * Step 1: Read Model DOC2VEC file and Re-build API vector 
		 * Name		= result_map
		 * Key		= API name e.g. java.util.collections
		 * Value	= Map < String Key, Double Value>
		 * 			Key = Tutorial ID e.g. DOC_0, DOC_1
		 * 			Value = cosine sim btw API and tutorial
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
	}
	
	public static void readOracleResult() throws IOException
	{
		/*
		 * Step 2:	Read Oracle file and build a Map
		 * Name		= oracle_map
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
	}
	
	public static void readFrapt() throws IOException
	{
		/*
		 * Read FRAPT result
		 */
		File FRAPT_result = new File("resources/input/FRAPT_result_col.txt");
		Scanner sc = new Scanner(FRAPT_result);
		while(sc.hasNextLine()){
			String s = sc.nextLine();
			s = s.replace(":", "");
			Scanner sc_in = new Scanner(s);
			while (sc_in.hasNext()){
				String DOC_ID = sc_in.next();
				String class_name = sc_in.next();
				int result = sc_in.nextInt();
				String key = DOC_ID + "_" + class_name;
				boolean value	= (result == 1) ? true : false;
				FRAPT_map.put(key, value);
			}
			sc_in.close();
		}
		sc.close();
	}
	
	public static void resultDoc2Vec() throws IOException
	{
		File file_output	= new File ("resources/output/result/"+model_name+"_Fscore");
		FileWriter fWriter	= new FileWriter (file_output);
		PrintWriter writer	= new PrintWriter (fWriter);
		/*
		 * Step 3: Precision & Recall & F1 with Threshold
		 */
		System.out.println("T\tPrecision\tRecall\tF\tTP");
		for (int j = 0; j <= 100; j++){
			int TP = 0, FP = 0, FN = 0;
			double T = j * 0.01;
			
			//int num_retrieve = 0;
			//			for (Entry<String,Map<String,Double>> entry : result_map.entrySet()){
			//				String class_name = entry.getKey();
			//				for (Entry<String,Double> entry_in : entry.getValue().entrySet()){
			//					if (entry_in.getValue() >= T){
			//						String key = entry_in.getKey().replaceAll("[^0-9]", "");
			//						key = key + "_" + class_name;
			//						if (oracle_map.containsKey(key)){
			//							if (oracle_map.get(key))
			//								TP ++;
			//							else
			//								FP ++;
			//						}else{
			//							FP ++;
			//						}
			//						num_retrieve ++;
			//					}
			//				}
			//			}
			
			for( Entry<String, Boolean> entry: oracle_map.entrySet() )
			{
				int index = entry.getKey().indexOf("_");
				String fragNo = entry.getKey().substring(0,index), className = entry.getKey().substring(index+1);

				if ( result_map.containsKey(className) )
				{
					double result = result_map.get(className).get("DOC_" + fragNo);
					if ( result >= T)
					{
						//We say 1, oracle say 1
						if( entry.getValue() ) 
						{
							TP++;
						}
						else //We say 1, oracle say 0
						{
							FN++;
						}
					}
					else
					{
						if( entry.getValue() ) //We say 0, oracle say 1
						{
							FP++;
						}
					}
				}

//				else
//				{
//					FN++;
//				}
				
			}
			//			double precision    = 100 * (double)TP / num_retrieve;
			double precision 	= 100 * (double)TP / (TP+FP);
			//			double recall 		= 100 * (double)TP / num_relevant_pair;
			double recall    	= 100 * (double)TP / (TP+FN);
			double F1			= 2 * precision * recall / (precision + recall);
			

			System.out.println(T+"\t"+precision+"\t"+recall + "\t" + F1 + "\t"+ TP);
			writer.println(T+" "+precision+" "+recall+" "+F1);

		}
		// *****************************************************************************
		writer.close();

	}
	
	public static void resultFraptIn() throws IOException
	{
		int TPf = 0, FPf = 0, FNf = 0; //For FRAPT accuracy
		for( Entry<String, Boolean> entry: oracle_map.entrySet() )
		{
			String className = entry.getKey().substring(entry.getKey().indexOf("_")+1);

			if ( result_map.containsKey(className) )
			{
				boolean fraptResult = FRAPT_map.get(entry.getKey());
				if ( entry.getValue() ) //Oracle say 1
				{
					if (fraptResult) TPf++; //Frapt say 1
					else FPf++; 			//Frapt say 0
				}
				else						//Oracle say 0
				{
					if (fraptResult) FNf++; //Frapt say 1
				}
			}
		}
		double precisionF 	= 100 * (double)TPf / (TPf+FPf);
		double recallF    	= 100 * (double)TPf / (TPf+FNf);
		double F1F			= 2 * precisionF * recallF / (precisionF + recallF);
		System.out.println("FRAPT result ------------------------");
		System.out.println(precisionF+" "+recallF+" "+F1F);

	}
	public static void topK() throws IOException
	{
		/*
		 * K-Accuracy
		 */
		/*		
		 * Initialize Log file
		 */
		File file_output1	= new File ("resources/output/result/"+model_name+"_Top_"+k_top+"_Accuracy");
		FileWriter fWriter1	= new FileWriter (file_output1);
		PrintWriter writer1	= new PrintWriter (fWriter1);
		double[] count_k = new double[k_top];	// Numerator Array
		for (Entry<String, Map<String, Double>> entry : result_map.entrySet()){
			//			System.out.println(entry.getKey());
			//			System.out.println(entry.getValue().toString());
			int j = 0;
			String class_name = entry.getKey();
			for (Entry<String,Double> entry_in : entry.getValue().entrySet()){
				if (j < k_top){
					String key = entry_in.getKey().replaceAll("[^0-9]", "");
					key = key + "_" + class_name;
					if (oracle_map.containsKey(key)){
						if (oracle_map.get(key)){
							for (int k = j; k < count_k.length; k++){
								count_k[k] ++;
							}
							break;
						}					
					}
				}else{
					break;
				}

				j++;
			}

		}
		System.out.println("Top " + k_top + " Accuracy");
		writer1.println("Top " + k_top + " Accuracy");
		for (int m = 0; m < count_k.length; m++){
			count_k[m] = count_k[m] * 100 / (31);
			//	    	  System.out.println("Top "+(num.length - j)+" accuracy:\n "+num[j]);
			System.out.println(count_k[m]);
			writer1.println(count_k[m]);
		}
		writer1.close();
		// *****************************************************************************
	}
}
