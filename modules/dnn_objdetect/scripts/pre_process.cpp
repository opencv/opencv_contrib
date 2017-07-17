#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string.h>

#include <rapidxml.hpp>

typedef struct {
   int width;
   int height;
}point;

void collect_data(const char *file_name, std::vector<point> &stats_list){
   std::fstream stats;
   stats.open(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
   std::string stats_line;
   point _point;

   while(std::getline(stats, stats_line)){
      std::vector<std::string> stats_per_line;
      std::string delim(" ");
      std::string token;
      int pos=0;

      while((pos = stats_line.find(delim)) != std::string::npos){
         token = stats_line.substr(0, pos);
         stats_per_line.push_back(token);
         stats_line.erase(0, pos+delim.length());
      }
      for(std::vector<std::string>::iterator itr=stats_per_line.begin()+1;
         itr != stats_per_line.end(); ++itr){
         int idx = itr - stats_per_line.begin();
         if(idx % 2 != 0){
            _point.width = stoi(*itr);
         }else{
            _point.height = stoi(*itr);
            stats_list.push_back(_point);
         }
      }
   }
   std::cout << "Processed total of " << stats_list.size() << " objects\n";
   stats.close();
}

int main(int argc, char **argv) {
   std::vector<std::string> file_names;
   int width, height;
   std::vector<std::pair<int, int>> bbox;
   const std::string bbox_ann("<bndbox>");
   std::map<std::string, int> obj_class_mapping;
   std::string class_names[] = {"aeroplane","bicycle","bird","boat","bottle",
                                "bus","car","cat","chair","cow","diningtable",
                                "dog","horse","motorbike","person","pottedplant",
                                "sheep","sofa","train","tvmonitor"};

   for (size_t i = 0; i < 20; ++i) {
     obj_class_mapping[class_names[i]] = i;
   }
   if(argc < 4) {
      std::cerr << "Usage: " << argv[0]
                << " <annotation_dir_name>"
                << " <new_annotation_dir_name>"
                << " <annotation_file_name>"
                << " <statistics_file_name>"
                << std::endl;
      return -1;
   }
   std::string dir_name(argv[1]);
   std::string dir_name_new(argv[2]);
   std::ifstream infile(argv[3]);
   std::string line;

   while(std::getline(infile, line))
      file_names.push_back(line);

   for(std::vector<std::string>::iterator itr = file_names.begin();
      itr != file_names.end(); ++itr) {
      std::ostringstream file_path;
      file_path << dir_name << "/" << *itr;
      std::fstream bbox_stats;
      bbox_stats.open(dir_name_new+"/"+*itr, std::fstream::out | std::fstream::app);

      std::ifstream single_parse(file_path.str().c_str());

      std::string xml_content, xml_line;
      while(std::getline(single_parse, xml_line))
         xml_content += xml_line;

      std::vector<char> xml_content_copy(xml_content.begin(), xml_content.end());
      xml_content_copy.push_back('\0');

      // Parse each file
      rapidxml::xml_document<> doc;
      doc.parse<0>(&xml_content_copy[0]);
      std::cout << "Processing file: " << file_path.str().c_str() << std::endl;
      rapidxml::xml_node<> *node = doc.first_node()->first_node("object");

      // Iterate through all the siblings of the node
      while(node && strcmp(node->name(), "object")==0){
         rapidxml::xml_node<> *child_node = node->first_node("bndbox");
         rapidxml::xml_node<> *child = node->first_node("name");

         const int class_id = obj_class_mapping[node->first_node("name")->value()];
         const int xmin = atoi(child_node->first_node("xmin")->value());
         const int xmax = atoi(child_node->first_node("xmax")->value());
         const int ymin = atoi(child_node->first_node("ymin")->value());
         const int ymax = atoi(child_node->first_node("ymax")->value());

         height = ymax - ymin;
         width  = xmax - xmin;

         bbox_stats << xmin << " " << ymin << " "
                    << xmax << " " << ymax << " "
                    << class_id << " ";
         bbox.push_back(std::make_pair(width, height));

         node = node->next_sibling();
      }
      bbox_stats.close();
   }
   std::cout << "Processed " << file_names.size() << " files\n";
   std::cout << "Total objects found: " << bbox.size() << std::endl;
   return 0;
}
