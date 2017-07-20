package model;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Map;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.stream.Collectors;

public class Model {




	private static final Logger log = LoggerFactory.getLogger(Model.class);
	
	public final static int num_FragTutorial 	= 57; 
	public final static int num_class 			= 31; 
	public final static int num_relevant_pair 	= 56; 
	public static double MAX = -1;
	public static double MIN = 1;
	
	
	public static Map<String,List<String>> training_map = new HashMap<>();
//	public static Map<String,Boolean> oracle_map = new HashMap<>();
	public static Map<String,Map<String,Double>> result_map = new HashMap<>();
	
	// Load training corpus
	public final static String model_name = "col_concatinate_max";
	public final static String path = "/tutorial_javadoc_col_concatinate.txt";
	
//	public static String path = "/collections_single.txt";
//	public static String path = "/tutorial_javadoc_col_combine2.txt";
	//		String path = "/description+1.txt";
	//		String path = "/description+2.txt";
	//    	String path = "/tutorial_javadoc_col_full.txt";
	//    	String path = "/tutorial_javadoc_col_full_short.txt";
	//    	String path = "/raw_sentences.txt";
	
	public static void main(String[] args) throws Exception {
		// Configure training Para2Vec
		ClassPathResource resource = new ClassPathResource("input"+path);
		File file_training_corpus = resource.getFile();
		SentenceIterator iter = new BasicLineIterator(file_training_corpus);

		AbstractCache<VocabWord> cache = new AbstractCache<>();

		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		LabelsSource source = new LabelsSource("DOC_");


		// Train Doc2Vec 
		ParagraphVectors vec = new ParagraphVectors.Builder()
				.minWordFrequency(1) //change from 1 to 5
				.iterations(5)	// change from 5 to 10
				.epochs(20)		//Change from 20 to 1
				.layerSize(100) // change from 100 to 200
				.learningRate(0.005) // change from 0.025 to 0.01
				.labelsSource(source)
				.windowSize(5) 	// change from 5 to 3
				.iterate(iter)
				.trainWordVectors(true) // change from false to true
				.vocabCache(cache)
				.tokenizerFactory(t)
				.sampling(0)
				.build();

		vec.fit();	// Start the training

		// *****************************************************************************

		/*		Initialize Log file
				Store info of training */

		File file_output = new File ("resources/output/model/"+model_name+"_norange");
		FileWriter fWriter = new FileWriter (file_output);
		PrintWriter writer = new PrintWriter (fWriter);
		
		File file_output1 = new File ("resources/output/model/"+model_name+"_range");
		FileWriter fWriter1 = new FileWriter (file_output1);
		PrintWriter writer1 = new PrintWriter (fWriter1);	
		
		// *****************************************************************************
		
		/*	
		 * 	Step 1: Read Training Corpus and build a Map
		 *  Key		= API name
		 *  Value	= List / Array of its DOC_ID
		*/
		Scanner sc_training_corpus = new Scanner(file_training_corpus);
		int i = 0;
		String first_token 	= "";
		String last_token	= "";
		List<String> DOC_ID = new ArrayList<String>();
		while (sc_training_corpus.hasNextLine()){
			String s = sc_training_corpus.nextLine();
			if (i > num_FragTutorial - 1){				
				String arr[] = s.split(" ", 2);
				first_token = arr[0].replace(":", "");	// Read the first token
	
				if (!first_token.equals(last_token)){

					DOC_ID = new ArrayList<String>();
				}
				DOC_ID.add("DOC_"+i);

				if (training_map.containsKey(first_token)){
					
					training_map.replace(first_token, DOC_ID);
				}else{
					training_map.put(first_token, DOC_ID);
					
				}
				
				last_token = first_token;	
			}
			i++;
		}
		sc_training_corpus.close();
		
		
		// *****************************************************************************
		
		/*
		 * Step 2: Build API vector using MAX / AVERAGE function
		 * Cosine similarity
		 */

		for (Entry<String, List<String>> entry : training_map.entrySet()){
			Map<String,Double> simi_map  = max(entry.getValue(),vec);
//			Map<String,Double> simi_map  = average(entry.getValue(),vec);
			
			simi_map	= sortByValue(simi_map);
			result_map.put(entry.getKey(), simi_map);
		}
		// *****************************************************************************
		
		/*
		 * Step 3: Export model
		 */
		for (Entry<String,Map<String,Double>> entry : result_map.entrySet()){
			System.out.println(entry.getKey());
			writer.println(entry.getKey());
			writer1.println(entry.getKey());
			for (Entry<String,Double> entry_in : entry.getValue().entrySet()){
				System.out.print(entry_in.getKey() + " " + entry_in.getValue() + " ");
				writer.print(entry_in.getKey() + " " + entry_in.getValue() + " ");
			}
			System.out.println();
			writer.println();
			
			entry.setValue(convert_range_map(entry.getValue(),MIN,MAX));
			for (Entry<String,Double> entry_in : entry.getValue().entrySet()){
//				System.out.print(entry_in.getKey() + " " + entry_in.getValue() + " ");
				writer1.print(entry_in.getKey() + " " + entry_in.getValue() + " ");
			}
			writer1.println();

		}
		fWriter.close();
		fWriter1.close();
		System.out.println("MIN: "+MIN);
		System.out.println("MAX: "+MAX);
		
		// *****************************************************************************
	
	}

	
	public static double convert_range(double x ,double in_min, double in_max, double out_min, double out_max){
		double new_x = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min; 
		return new_x;
		
	}
	public static Map<String,Double> convert_range_map(Map<String,Double> map, double range_min, double range_max){
		for (Entry<String,Double> entry : map.entrySet()){
			double new_simi = convert_range(entry.getValue(),range_min,range_max,0,1);
			entry.setValue(new_simi);
		}
		return map;
	}
	
//	public static Map<String,Double> convert(Map<String,Double> map){
//		double min = 1;
//		double max = -1;
//		for (Entry<String,Double> entry : map.entrySet()){
//			double value = entry.getValue();
//			if (max < value)
//				max = value;
//			if (min > value)
//				min = value;
//
//		}
//		
//		
//		
//		return convert_range_map(map,min,max);
//	}
	
	public static Map<String,Double> max(List<String> list,ParagraphVectors vec){
		/*
		 * Key 	 = DOC ID
		 * Value = Cosine Similarity
		 */
		Map<String,Double> simi_map  = new TreeMap<>();
		
		for (String ID : list){
			for (int i = 0; i < num_FragTutorial; i++){
				double new_simi	= vec.similarity(ID, "DOC_"+i);
				double current_simi	= simi_map.getOrDefault("DOC_"+i, 2.0);
        		if (current_simi != 2.0 && current_simi < new_simi){
        			simi_map.replace("DOC_"+i, new_simi);

        		}else if(current_simi == 2.0){
        			simi_map.put("DOC_"+i, new_simi); // Put DOC_ID and Similarity 
        		} 			
			}
		}	
		
		return simi_map;
	}
	
	public static Map<String,Double> average(List<String> list,ParagraphVectors vec){
		Map<String,Double> simi_map  = new HashMap<>();
		for (String ID : list){
			for (int i = 0; i < num_FragTutorial; i++){
				String TUT_ID = "DOC_"+i;
				double new_simi	= vec.similarity(ID, TUT_ID);
				double current_simi	= simi_map.getOrDefault(TUT_ID, 2.0);
        		if (current_simi != 2.0){
        			new_simi = new_simi + simi_map.get(TUT_ID);
        			simi_map.replace(TUT_ID, new_simi);
        		}else if(current_simi == 2.0){
        			simi_map.put(TUT_ID, new_simi); // Put DOC_ID and Similarity 
        		}				
			}
		}
//		System.out.println("Size: "+list.size());
		for (Entry<String,Double>entry : simi_map.entrySet()){
//			System.out.println(entry.getValue());
			double value = entry.getValue() / list.size();
//			training_map.get(key)
			entry.setValue(value);
//			System.out.println(entry.getValue());
		}
		
		return simi_map;
	}
	
	// Sort Map by Value
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
		Map<K, V> m = map.entrySet()
				.stream()
				.sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
				.collect(Collectors.toMap(
						Map.Entry::getKey, 
						Map.Entry::getValue, 
						(e1, e2) -> e1, 
						LinkedHashMap::new
						));
		for (Entry<K,V> entry : m.entrySet()){
			double value = (double) entry.getValue();
			if (MAX < value)
				MAX = value;
			if (MIN > value)
				MIN = value;
		}
		return m;
//		return map.entrySet()
//				.stream()
//				.sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
//				.collect(Collectors.toMap(
//						Map.Entry::getKey, 
//						Map.Entry::getValue, 
//						(e1, e2) -> e1, 
//						LinkedHashMap::new
//						));
	}
	
	
}
