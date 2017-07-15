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
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * This is example code for dl4j ParagraphVectors implementation. In this example we build distributed representation of all sentences present in training corpus.
 * However, you still use it for training on labelled documents, using sets of LabelledDocument and LabelAwareIterator implementation.
 *
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class Fragmentation {
	
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
	    return map.entrySet()
	              .stream()
	              .sorted(Map.Entry.comparingByValue(/*Collections.reverseOrder()*/))
	              .collect(Collectors.toMap(
	                Map.Entry::getKey, 
	                Map.Entry::getValue, 
	                (e1, e2) -> e1, 
	                LinkedHashMap::new
	              ));
	}
	
	   private static void printArray(int[] anArray) {
		      for (int i = 0; i < anArray.length; i++) {
		         if (i > 0) {
		            System.out.print(", ");
		         }
		         System.out.print(anArray[i]);
		      }
		   }

    private static final Logger log = LoggerFactory.getLogger(Fragmentation.class);

    public static void main(String[] args) throws Exception {
//    	String path = "/tutorial_javadoc_col_concatinate.txt";
//    	String path = "/collections_single.txt";
//    	String path = "/tutorial_javadoc_col_combine2.txt";
    	String path = "/description+1.txt";
//    	String path = "/tutorial_javadoc_col_combine";
//    	String path = "/tutorial_javadoc_col_full.txt";
//    	String path = "/tutorial_javadoc_col_full_short.txt";
//    	String path = "/raw_sentences.txt";
        ClassPathResource resource = new ClassPathResource(path);
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        AbstractCache<VocabWord> cache = new AbstractCache<>();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        /*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.

              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
        */
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
                .trainWordVectors(false) // change from false to true
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();
        
        vec.fit();
        
//        PrintWriter writer = new PrintWriter("log.txt", "UTF-8");
//        ClassPathResource resource_out = new ClassPathResource("/log.txt");
//        File outFile = resource_out.getFile();
        File outFile = new File ("E:/output.txt");
        FileWriter fWriter = new FileWriter (outFile);
        PrintWriter writer = new PrintWriter (fWriter);
        writer.println ("This is a line.");
//        pWriter.println ("This is another line.");
//        pWriter.close();
    	
        // Read Oracle file and build a Map for APIs    	
    	// Build an Array represented Oracle results
        ClassPathResource resource1 = new ClassPathResource("/tutorial_col_oracle");
        File file1 = resource1.getFile();
        Scanner oracle_file = new Scanner(file1);
        
        int index = 0;
//        int[] oracle = new int[tutorial_endline];
        Map<String,int[]> API_map = new TreeMap<>();
        
        while (oracle_file.hasNextLine()){
        	Scanner line = new Scanner(oracle_file.nextLine());
        	while (line.hasNext()){
        		String API_name = line.next();
    			int result = Integer.parseInt(line.next());
    			int[] temp = new int[57];
        		if (API_map.containsKey(API_name)){
//        			int result = Integer.parseInt(line.next());
        			temp = API_map.get(API_name);
        			temp[index] = result;
        			API_map.replace(API_name, temp);
        		}else{
        			temp[index] = result;
        			API_map.put(API_name, temp);
        		}
        	}
        	index++;
        }
        oracle_file.close();
        
        // Print the Map
//        for (Entry<String, int[]> entry : API_map.entrySet()){
//        	System.out.println(entry.getKey());
//        	printArray(entry.getValue());
//        	 System.out.println();
//        }
        
        
        
        Map<String, Double> map = new TreeMap<>();
        Scanner sc = new Scanner(file); // training file
        int i = 0;
    	int numerator = 0;
//    	int denominator = 1581; // 58 -> 1638 = 1581
//    	int denominator = 88;
//    	int denominator = 623;
    	int denominator = 31;
//    	int denominator = 996;
    	int tutorial_endline 	= 57;
    	int javadoc_endline		= 145;
    	int k_nearest = 7; 
    	double[] num = new double[k_nearest];	// Numerator Array
    	String first_token ="";
    	String last_token ="";
//    	Double default_value = 2;
        while(sc.hasNextLine()){
        	
        	// Skip tutorial
        	if (i > 56){
        		first_token = sc.next().replace(":", "");	// Read the first token
//        		System.out.println(first_token);
        		if (!first_token.equals(last_token)){	// New Class
            		map = sortByValue(map);	// Sort the map
            		
                	int it = 0;              	
                	
                	// Top k accuracy
                	// Iterate through the sorted map and find the top k elements
            		int index1 = 0;
            		int index2 = 0;
                	for (Entry<String, Double> entry : map.entrySet())
                	{
                		
                		// Only consider k last elements in the sorted map
                		if (it >= (tutorial_endline - k_nearest)){
//                			System.out.println("token: "+first_token);
                			System.out.print(entry.getKey() + ": " + entry.getValue());
                    		String number = entry.getKey().replaceAll("[^0-9]", "");
                    		int[] temp = API_map.get(last_token);
                    		int oracle_result = temp[Integer.parseInt(number)];
                    	
                    		System.out.print("\t"+oracle_result+"\t");
                    		                		
                    		if (oracle_result == 1){
//                    			System.out.print("\tMatch");
                    			while(index1 >= index2){
                    				System.out.print("Plus top "+(num.length - index2));
                    				num[index2] ++;
                    				index2 ++;
                    			}
                    		}
                    		index1 ++;
                		}            			
                	    it++;
                	}
        			map.clear();
        			System.out.println();
        		}
        		
        		// Calculate the cosine similarity and put / replace the Map
        		
        		// For each fragment in tutorials, cosine similarity between javadoc fragment and tutorial fragments
        		// i: javadoc and j: tutorial fragments
        		for (int j = 0; j < 57; j++){
            		double sim = vec.similarity("DOC_"+i, "DOC_"+j);	// Cosine similarity btw DOC_57 and DOC_0-56
            		double current_sim = map.getOrDefault("DOC_"+j, 2.0);
            		writer.print("DOC_"+i+"\tDOC_"+j+":\t");
            		writer.print("current sim: "+current_sim+" new sim: "+sim);
//            		System.out.print("current sim: "+current_sim+"\tnew sim: "+sim);
//            		System.out.print("current sim: "+current_sim+"\tnew sim: "+sim);
            		if (current_sim != 2.0 && current_sim < sim){
            			map.replace("DOC_"+j, sim);
            			writer.print("\tREPLACE");
            		}else if(current_sim == 2.0){
            			writer.print("\tPUT");
            			map.put("DOC_"+j, sim); // Put DOC_ID and Similarity 
            		}
            		writer.println();
            		     			
        		}
        		
        		last_token = first_token;
            	
        	}
        	
        	sc.nextLine();
        	i++;
        }
//      denominator = i - tutorial_endline;
        denominator = 31;
//      System.out.println("Denominator: "+(i-tutorial_endline));
      for (int j = num.length-1; j>=0; j--){
    	  num[j] = num[j] * 100 / denominator;
//    	  System.out.println("Top "+(num.length - j)+" accuracy:\n "+num[j]);
    	  System.out.println(num[j]);
      }
      
//      log.info("Plot TSNE....");
//      BarnesHutTsne tsne = new BarnesHutTsne.Builder()
//              .setMaxIter(1000)
//              .stopLyingIteration(250)
//              .learningRate(500)
//              .useAdaGrad(false)
//              .theta(0.5)
//              .setMomentum(0.5)
//              .normalize(true)
//              .usePca(false)
//              .build();
//      vec.lookupTable().plotVocab(tsne);
//      for (Entry<String, int[]> entry : API_map.entrySet()){
//      	System.out.println(entry.getKey());
//      	printArray(entry.getValue());
//      	 System.out.println();
//      }
//      double accu = (double)numerator * 100 / denominator;
//      System.out.println("Numerator: "+numerator);
//      System.out.println("denominator: "+denominator);
//      System.out.println("Accuracy: "+accu);
      
//        while(file.)
        
        
        // K-nearest Calculation
//    	int numerator = 0;
//    	int denominator = javadoc_endline-tutorial_endline;
//        for (int i = tutorial_endline; i < javadoc_endline ; i++){
//        	for (int j = 0; j < tutorial_endline; j++){
//        		double sim = vec.similarity("DOC_"+i, "DOC_"+j);
//        		map.put("DOC_"+j, sim);
//        	}
//        	map = sortByValue(map);
//        	
//        	int it = 0;
//        	
//
//        	for (Entry<String, Double> entry : map.entrySet())
//        	{
//        		
//        		
//        		if (it >= (tutorial_endline - k_nearest)){
//        			System.out.print(entry.getKey() + ": " + entry.getValue());
//            		String number = entry.getKey().replaceAll("[^0-9]", "");
//            		int oracle_result = oracle[Integer.parseInt(number)];
//            		System.out.print("\t"+oracle_result+"\t");
//            		if (oracle_result == 1){
//            			numerator ++;
//            			System.out.print("\tMatch");
//            			break;
//            		}
//        		}
//        			
//        	    it++;
//        	}
//        	System.out.println();
//
//        }
//        
//        double accu = (double)numerator * 100 / denominator;
//        System.out.println("Accuracy: "+accu);

        
        
        
        
        
//        String s = "DOC_20";
//        System.out.println( s.replaceAll("[^0-9]", ""));
       
        
//        // Tutorial
//        // http://docs.oracle.com/javase/tutorial/collections/algorithms/index.html
//        String para1 = "The sort algorithm reorders a List so that its elements are in ascending order according to an ordering relationship. Two forms of the operation are provided. The simple form takes a List and sorts it according to its elements' natural ordering. If you're unfamiliar with the concept of natural ordering, read the Object Ordering section.  The sort operation uses a slightly optimized merge sort algorithm that is fast and stable:  Fast: It is guaranteed to run in n log(n) time and runs substantially faster on nearly sorted lists. Empirical tests showed it to be as fast as a highly optimized quicksort. A quicksort is generally considered to be faster than a merge sort but isn't stable and doesn't guarantee n log(n) performance. Stable: It doesn't reorder equal elements. This is important if you sort the same list repeatedly on different attributes. If a user of a mail program sorts the inbox by mailing date and then sorts it by sender, the user naturally expects that the now-contiguous list of messages from a given sender will (still) be sorted by mailing date. This is guaranteed only if the second sort was stable. The following trivial program prints out its arguments in lexicographic (alphabetical) order.   import java.util.*;  public class Sort {     public static void main(String[] args) {         List<String> list = Arrays.asList(args);         Collections.sort(list);         System.out.println(list);     } } Let's run the program.  % java Sort i walk the line The following output is produced.  [i, line, the, walk] The program was included only to show you that algorithms really are as easy to use as they appear to be.  The second form of sort takes a Comparator in addition to a List and sorts the elements with the Comparator. Suppose you want to print out the anagram groups from our earlier example in reverse order of size — largest anagram group first. The example that follows shows you how to achieve this with the help of the second form of the sort method.  Recall that the anagram groups are stored as values in a Map, in the form of List instances. The revised printing code iterates through the Map's values view, putting every List that passes the minimum-size test into a List of Lists. Then the code sorts this List, using a Comparator that expects List instances, and implements reverse size-ordering. Finally, the code iterates through the sorted List, printing its elements (the anagram groups). The following code replaces the printing code at the end of the main method in the Anagrams example.  // Make a List of all anagram groups above size threshold. List<List<String>> winners = new ArrayList<List<String>>(); for (List<String> l : m.values())     if (l.size() >= minGroupSize)         winners.add(l);  // Sort anagram groups according to size Collections.sort(winners, new Comparator<List<String>>() {     public int compare(List<String> o1, List<String> o2) {         return o2.size() - o1.size();     }});  // Print anagram groups. for (List<String> l : winners)     System.out.println(l.size() ++ l); Running the program on the same dictionary as in The Map Interface section, with the same minimum anagram group size (eight), produces the following output.";
//        
//        // Javadoc
//        // https://docs.oracle.com/javase/7/docs/api/java/util/Collections.html#sort(java.util.List)
//        String para2 = "sort public static <T extends Comparable<? super T>> void sort(List<T> list) Sorts the specified list into ascending order, according to the natural ordering of its elements. All elements in the list must implement the Comparable interface. Furthermore, all elements in the list must be mutually comparable (that is, e1.compareTo(e2) must not throw a ClassCastException for any elements e1 and e2 in the list). This sort is guaranteed to be stable: equal elements will not be reordered as a result of the sort.  The specified list must be modifiable, but need not be resizable.  Implementation note: This implementation is a stable, adaptive, iterative mergesort that requires far fewer than n lg(n) comparisons when the input array is partially sorted, while offering the performance of a traditional mergesort when the input array is randomly ordered. If the input array is nearly sorted, the implementation requires approximately n comparisons. Temporary storage requirements vary from a small constant for nearly sorted input arrays to n/2 object references for randomly ordered input arrays.  The implementation takes equal advantage of ascending and descending order in its input array, and can take advantage of ascending and descending order in different parts of the same input array. It is well-suited to merging two or more sorted arrays: simply concatenate the arrays and sort the resulting array.  The implementation was adapted from Tim Peters's list sort for Python ( TimSort). It uses techiques from Peter McIlroy's Optimistic Sorting and Information Theoretic Complexity, in Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms, pp 467-474, January 1993.  This implementation dumps the specified list into an array, sorts the array, and iterates over the list resetting each element from the corresponding position in the array. This avoids the n2 log(n) performance that would result from attempting to sort a linked list in place.  Parameters: list - the list to be sorted.";
//        
//        // Finance news
//        String para3 = "The LSE allows companies to raise money, increase their profile and obtain a market valuation through a variety of routes, thus following the firms throughout the whole IPO process.";
//        
//        // Wikipedia cosine similarity definition
//        String para4 = "Although the term cosine similarity has been used for this angular distance, the term is used as the cosine of the angle only as a convenient mechanism for calculating the angle itself and is no part of the meaning. The advantage of the angular similarity coefficient is that, when used as a difference coefficient (by subtracting it from 1) the resulting function is a proper distance metric, which is not the case for the first meaning. However, for most uses this is not an important property. For any use where only the relative ordering of similarity or distance within a set of vectors is important, then which function is used is immaterial as the resulting order will be unaffected by the choice.";
//        
//        String para5 = "Collections";
//        String para6 = "Lists";
//        
//        // Tutorial
//        // http://docs.oracle.com/javase/tutorial/collections/algorithms/index.html
//        String para7 = "Searching  The binarySearch algorithm searches for a specified element in a sorted List. This algorithm has two forms. The first takes a List and an element to search for (the search key). This form assumes that the List is sorted in ascending order according to the natural ordering of its elements. The second form takes a Comparator in addition to the List and the search key, and assumes that the List is sorted into ascending order according to the specified Comparator. The sort algorithm can be used to sort the List prior to calling binarySearch.  The return value is the same for both forms. If the List contains the search key, its index is returned. If not, the return value is (-(insertion point) - 1), where the insertion point is the point at which the value would be inserted into the List, or the index of the first element greater than the value or list.size() if all elements in the List are less than the specified value. This admittedly ugly formula guarantees that the return value will be >= 0 if and only if the search key is found. It's basically a hack to combine a boolean (found) and an integer (index) into a single int return value.  The following idiom, usable with both forms of the binarySearch operation, looks for the specified search key and inserts it at the appropriate position if it's not already present.  int pos = Collections.binarySearch(list, key); if (pos < 0)    l.add(-pos-1, key);";
//        
//        // Javadoc
//        // https://docs.oracle.com/javase/7/docs/api/java/util/Collections.html#binarySearch(java.util.List,%20T)
//        String para8 = "binarySearch public static <T> int binarySearch(List<? extends Comparable<? super T>> list,                    T key) Searches the specified list for the specified object using the binary search algorithm. The list must be sorted into ascending order according to the natural ordering of its elements (as by the sort(List) method) prior to making this call. If it is not sorted, the results are undefined. If the list contains multiple elements equal to the specified object, there is no guarantee which one will be found. This method runs in log(n) time for a random access list (which provides near-constant-time positional access). If the specified list does not implement the RandomAccess interface and is large, this method will do an iterator-based binary search that performs O(n) link traversals and O(log n) element comparisons.  Parameters: list - the list to be searched. key - the key to be searched for. Returns: the index of the search key, if it is contained in the list; otherwise, (-(insertion point) - 1). The insertion point is defined as the point at which the key would be inserted into the list: the index of the first element greater than the key, or list.size() if all elements in the list are less than the specified key. Note that this guarantees that the return value will be >= 0 if and only if the key is found.";
//        
//        INDArray inferredVector1 = vec.inferVector(para1);
//        INDArray inferredVector2= vec.inferVector(para2);
//        INDArray inferredVector3= vec.inferVector(para3);
//        INDArray inferredVector4= vec.inferVector(para4);
//        INDArray inferredVector5= vec.inferVector(para5);
//        INDArray inferredVector6= vec.inferVector(para6);
//        INDArray inferredVector7= vec.inferVector(para7);
//        INDArray inferredVector8= vec.inferVector(para8);
//        
//        log.info("Cosine similarity 1/2: tutorial - javadoc {}", Transforms.cosineSim(inferredVector1, inferredVector2));
//        log.info("Cosine similarity 1/3: tutorial - finance {}", Transforms.cosineSim(inferredVector1, inferredVector3));
//        log.info("Cosine similarity 2/3: javadoc  - finance {}", Transforms.cosineSim(inferredVector2, inferredVector3));
//        log.info("Cosine similarity 1/4: tutorial - wiki {}", Transforms.cosineSim(inferredVector1, inferredVector4));
//        log.info("Cosine similarity 1/5: tutorial - API  {}", Transforms.cosineSim(inferredVector1, inferredVector5));
//        log.info("Cosine similarity 1/6: tutorial - API  {}", Transforms.cosineSim(inferredVector1, inferredVector6));
//        log.info("Cosine similarity 7/8: tutorial - javadoc {}", Transforms.cosineSim(inferredVector7, inferredVector8));

        
     
//      double similarity1 = vec.similarity("DOC_1", "DOC_50");
//      log.info("2/51 similarity: (expect very high) " + similarity1);
//      
//	  double similarity2 = vec.similarity("DOC_1", "DOC_51");
//	  log.info("2/52 similarity: (expect high) " + similarity2);
//	  
////	    double similarity3 = vec.similarity("DOC_50", "DOC_51");
////	    log.info("51/52 similarity: " + similarity3);
//	    
//	    double similarity4 = vec.similarity("DOC_4", "DOC_52");
//	    log.info("5/53 similarity: (expect high) " + similarity4);    
//        vec.similarityToLabel(document, label)

//        double similarity = vec.similarityToLabel(document, label);
//        Log.info("similarity: "+similarity);
//        
//        double similarity1 = vec.similarityToLabel(para1, "DOC_1");
//        Log.info("similarity: "+similarity1);
//        log.info("Closest Words:");
//        Collection<String> lst = vec.wordsNearest(para1, 1);
//        System.out.println(lst);
        
        
//        double similarity1 = vec.similarity("DOC_1", "DOC_50");
//        log.info("1/50 similarity: " + similarity1);
//        
//        
//        String w = "binarySearch";
//	    Collection<String> nearest = vec.nearestLabels(w, 5);
//        System.out.println("top 5 of "+w+" :"+nearest);
//        
//        String w1 = "Collections";
//	    Collection<String> nearest1 = vec.nearestLabels(w1, 5);
//        System.out.println("top 5 of "+w1+" :"+nearest1);    
//
//        String w2 = "Lists";
//	    Collection<String> nearest2 = vec.nearestLabels(w2, 5);
//        System.out.println("top 5 of "+w2+" :"+nearest2);  
;
//
//        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
//        log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2);
//
//        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
//        log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3);
//
//        // likelihood in this case should be significantly lower
//        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
//        log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
//            "(should be significantly lower)");
    }
}
