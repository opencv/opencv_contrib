require 'erb'
require 'delegate'
require 'fileutils'

module Rbind
   class GeneratorC

       class HelperBase
           attr_accessor :includes
           attr_accessor :name

           def initialize(name,root)
               @root = root
               @name = name
               @includes = []
           end

           def wrap_includes
               includes.map do |i|
                   if i =~ /<.*>/
                       "#include #{i}"
                   else
                       "#include \"#{i}\""
                   end
               end.join("\n")
           end

           def binding
               Kernel.binding
           end
       end

       class TypesHelperHDR < HelperBase
           def initialize(name, root,extern_types)
               raise "wrong type #{root}" unless root.is_a? RDataType
               super(name,root)
               @type_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","type_wrapper.h")).read,nil,"-")
               @extern_types = extern_types
           end

           def cdelete_method
               @root.cdelete_method
           end

           def type_wrapper(t)
               @type_wrapper.result(t.binding)
           end

           def wrap_types
               str = ""
               @extern_types.each do |type|
                   next if type.basic_type?
                   str += type_wrapper(type)
               end
               @root.each_type do |type|
                   next if type.basic_type?
                   str += type_wrapper(type)
               end
               str
           end
       end

       class TypesHelper < HelperBase
           def initialize(name, root,extern_types)
               super(name,root)
               @extern_types = extern_types
               @type_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","type_delete.h")).read)
           end

           def cdelete_method
               @root.cdelete_method
           end

           def type_wrapper(t)
               @type_wrapper.result(t.binding)
           end

           def wrap_types
               str = ""
               @extern_types.each do |type|
                   next if type.basic_type?
                   str += type_wrapper(type)
               end
               @root.each_type do |type|
                   next if type.basic_type?
                   str += type_wrapper(type)
               end
               str
           end
       end

       class ConstsHelper < HelperBase
           def wrap_consts
               str = ""
               @root.each_container do |type|
                   str2 = ""
                   type.each_const(false) do |c|
                       str2 += "#{c.signature};\n"
                   end
                   if !str2.empty?
                        str += "\n\n//constants for #{type.full_name}\n"
                        str += str2
                   end
                   str
               end
               str
           end
       end

       class ConversionsHelperHDR < HelperBase
           def initialize(name,root,extern_types)
               super(name,root)
               @extern_types = extern_types
               @type_conversion = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","type_conversion.hpp")).read,nil,'-')
               @type_typedef = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","type_typedef.h")).read)
           end

           def type_conversion(t)
               @type_conversion.result(t.binding)
           end

           def type_typedef(t)
               @type_typedef.result(t.binding)
           end

           def wrap_conversions
               str = ""
               @extern_types.each do |type|
                   str += type_typedef(type) if type.typedef?
                   next if type.basic_type?
                   str += type_conversion(type)
               end
               @root.each_type do |type|
                   str += type_typedef(type) if type.typedef?
                   next if type.basic_type?
                   str += type_conversion(type)
               end
               str
           end
       end

       class ConversionsHelper < HelperBase
           def initialize(name,root,extern_types)
               super(name,root)
               @type_conversion = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","type_conversion.cc")).read,nil,'-')
               @extern_types = extern_types
           end

           def type_conversion(t)
               @type_conversion.result(t.binding)
           end

           def wrap_conversions
               str = ""
               @extern_types.each do |type|
                   next if type.basic_type?
                   str += type_conversion(type)
               end
               @root.each_type do |type|
                   next if type.basic_type?
                   str += type_conversion(type)
               end
               str
           end
       end

       class OperationsHDRHelper < HelperBase
           def initialize(name,root)
               super
           end

           def wrap_operations
               str = ""
               @root.each_container do |type|
                   str2 = ""
                   type.each_operation do |op|
                       str2 += "RBIND_EXPORTS #{op.csignature};\n"
                   end
                   if !str2.empty?
                       str += "\n\n///methods for #{type.full_name}\n"
                       str += str2
                   end
               end
               str
           end
       end

       class OperationsHelper < HelperBase
           class OperationHelper < SimpleDelegator
               def wrap_parameters
                   cparameters.map do |arg|
                       next if arg.type.basic_type?
                       "#{arg.type.to_single_ptr.signature} #{arg.name}_ = fromC(#{arg.name});\n\t"
                   end.compact.join("")
               end

               def wrap_call
                   paras = parameters.map do |arg|
                       "#{"*" if (!arg.type.ptr? && !arg.type.basic_type?)}#{arg.name}#{"_" if !arg.type.basic_type?}"
                   end.join(", ")
                   fct = if attribute?
                             if return_type.name == "void" && !return_type.ptr?
                                 "rbind_obj_->#{attribute.name} = #{paras};"
                             else
                                 if return_type.basic_type?
                                     "return rbind_obj_->#{attribute.name};"
                                 elsif return_type.ptr?
                                     "return toC(rbind_obj_->#{attribute.name},false);"
                                 else
                                     "return toC(&rbind_obj_->#{attribute.name},false);"
                                 end
                             end
                         else
                             fct = if !constructor? && (return_type.name != "void" || return_type.ptr?)
                                       # operator+, operator++ etc
                                       if operator? && parameters.size == 1
                                           if return_type.basic_type?
                                               "return *rbind_obj_ #{operator} #{paras};"
                                           elsif return_type.ref?
                                               # the returned value is not the owner of the object
                                               "return toC(&(*rbind_obj_ #{operator} #{paras}),false);"
                                           else
                                               "return toC(new #{return_type.full_name}(*rbind_obj_ #{operator} #{paras}));"
                                           end
                                       elsif return_type.basic_type?
                                           "return #{full_name}(#{paras});"
                                       elsif return_type.ptr?
                                           "return toC(#{full_name}(#{paras}),#{return_type.ownership?});"
                                       elsif return_type.ref?
                                           # the returned value is never owner of the memory
                                           "return toC(&#{full_name}(#{paras}),false);"
                                       else
                                           "return toC(new #{return_type.full_name}(#{full_name}(#{paras})));"
                                       end
                                   else
                                       if constructor?
                                           "return toC(new #{namespace}(#{paras}));"
                                       else
                                           if operator?
                                               "*rbind_obj_ #{operator} #{paras};"
                                           else
                                               "#{full_name}(#{paras});"
                                           end
                                       end
                                   end
                             #convert call to member call 
                             if instance_method?
                                 #add base class name space
                                 if((inherit? && !abstract?) || ambiguous_name?)
                                     fct.gsub(full_name,"rbind_obj_->#{base_class.name}::#{name}")
                                 else
                                     fct.gsub(full_name,"rbind_obj_->#{name}")
                                 end
                             else
                                 fct
                             end
                         end
               end

               def binding
                   Kernel.binding
               end
           end


           def initialize(name,root)
               super
               @operation_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","operation_wrapper.cc")).read,nil,"-")
           end

           def operation_wrapper(op)
               raise unless op
               op2 = OperationHelper.new(op)
               @operation_wrapper.result(op2.binding)
           end

           def wrap_operations
               str = ""
               @root.each_container do |type|
                   type.each_operation do |op|
                       str += operation_wrapper(op)
                   end
               end
               str
           end
       end

       class CMakeListsHelper < HelperBase
           def initialize(name,lib_name,pkg_config=Array.new,libs=Array.new,ruby_path)
               super(name,pkg_config)
               @libs = libs
               @library_name = lib_name
               @find_package = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","find_package.txt")).read)
               @ruby_path = ruby_path
           end

           def ruby_path
               @ruby_path
           end

           def find_packages
               @root.map do |pkg|
                    @find_package.result(pkg.instance_eval("binding"))
               end.join("")
           end

           def libs
               str = @root.map do |pkg|
                   "${#{pkg.upcase}_LIBS} ${#{pkg.upcase}_LDFLAGS_OTHER}"
               end.join(" ")
               str += " " + @libs.join(" ")
           end

           def library_name
               @library_name
           end
       end

       attr_accessor :includes
       attr_accessor :library_name
       attr_accessor :libs
       attr_accessor :pkg_config
       attr_accessor :generate_cmake
       attr_accessor :output_path
       attr_accessor :ruby_path

       def initialize(root,name,library_name)
           raise "wrong type #{root}" unless root.is_a? RNamespace
           @root = root
           @erb_types_hdr = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","types.h")).read)
           @erb_types = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","types.cc")).read)
           @erb_consts = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","consts.h")).read)
           @erb_operations = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","operations.cc")).read)
           @erb_operations_hdr = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","operations.h")).read)
           @erb_conversions = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","conversions.cc")).read)
           @erb_conversions_hdr = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","conversions.hpp")).read)
           @erb_cmakelists = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","CMakeLists.txt")).read)
           @erb_find_package = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","c","find_package.txt")).read)
           @includes = Array.new
           @pkg_config= Array.new
           @library_name = library_name
           @name = name
           @generate_cmake = true
           @libs = []
       end

       def generate(path = @output_path,ruby_path = @ruby_path)
           @output_path = path
           @ruby_path = ruby_path
           FileUtils.mkdir_p(path) if path && !File.directory?(path)
           file_types_hdr = File.new(File.join(path,"types.h"),"w")
           file_types = File.new(File.join(path,"types.cc"),"w")
           file_consts = File.new(File.join(path,"constants.h"),"w")
           file_operations = File.new(File.join(path,"operations.cc"),"w")
           file_operations_hdr = File.new(File.join(path,"operations.h"),"w")
           file_conversions = File.new(File.join(path,"conversions.cc"),"w")
           file_conversions_hdr = File.new(File.join(path,"conversions.hpp"),"w")

           # mark all extern types which must be wrapped
           extern_types = @root.used_extern_types

           types_hdr = TypesHelperHDR.new("_#{library_name.upcase}_TYPES_H_",@root,extern_types)
           file_types_hdr.write @erb_types_hdr.result(types_hdr.binding)

           types = TypesHelper.new("types",@root,extern_types)
           file_types.write @erb_types.result(types.binding)

           consts = ConstsHelper.new("_#{library_name.upcase}_CONSTS_H_",@root)
           file_consts.write @erb_consts.result(consts.binding)

           conversions_hdr = ConversionsHelperHDR.new("#{library_name.upcase}_CONVERSIONS_H_",@root,extern_types)
           conversions_hdr.includes += includes
           file_conversions_hdr.write @erb_conversions_hdr.result(conversions_hdr.binding)

           conversions = ConversionsHelper.new("conversions",@root,extern_types)
           file_conversions.write @erb_conversions.result(conversions.binding)

           operations_hdr = OperationsHDRHelper.new("_#{library_name.upcase}_OPERATIONS_H_",@root)
           file_operations_hdr.write @erb_operations_hdr.result(operations_hdr.binding)

           operations = OperationsHelper.new("operations",@root)
           file_operations.write @erb_operations.result(operations.binding)

           if generate_cmake && !File.exist?(File.join(path,"CMakeLists.txt"))
               file_cmakelists = File.new(File.join(path,"CMakeLists.txt"),"w")
               cmakelists = CMakeListsHelper.new(@name,@library_name,@pkg_config,@libs,@ruby_path)
               file_cmakelists.write @erb_cmakelists.result(cmakelists.binding)
               src_path = File.join(File.dirname(__FILE__),"templates","c","cmake")
               cmake_path = File.join(path,"cmake")
               FileUtils.mkdir_p(cmake_path) if !File.directory?(cmake_path)
               FileUtils.copy(File.join(src_path,"FindRuby.cmake"),File.join(cmake_path,"FindRuby.cmake"))
           end
       end
   end
end
