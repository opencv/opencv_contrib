# Template for opencv's new smart pointer
class OpenCVPtr2 < Rbind::RTemplateClass
    def initialize(name="cv::Ptr")
        super
    end

    def specialize(klass,*parameters)
        if parameters.size != 1
            raise ArgumentError,"OpenCVPtr2 does only support one template parameter. Got: #{parameters}}"
        end
        ptr_type = parameters.first

        klass.add_operation Rbind::ROperation.new(klass.name,nil,Rbind::RParameter.new("other",klass))
        klass.add_operation Rbind::ROperation.new(klass.name,nil,Rbind::RParameter.new("p",ptr_type.to_ptr))
        klass.add_operation Rbind::ROperation.new("release",type("void"))
        klass.add_operation Rbind::ROperation.new("reset",type("void"),Rbind::RParameter.new("p",ptr_type.to_ptr))
        klass.add_operation Rbind::ROperation.new("swap",type("void"),Rbind::RParameter.new("other",klass))
        klass.add_operation Rbind::ROperation.new("get",ptr_type.to_ptr.to_ownership(false))
        klass.add_operation Rbind::ROperation.new("empty",type("bool"))
        klass
    end

    def specialize_ruby_specialization(klass)
        "    def method_missing(m,*args)\n"\
            "        raise \"Ptr #{self} is empty. Cannot call \#{m} on it!\" if empty\n"\
        "        get.method(m).call(*args)\n"\
            "    end\n"
    end
end

class Vec < Rbind::RClass
    def initialize(name,type,size)
        super(name)
        add_attribute Rbind::RAttribute.new("val",type.to_ptr)
        add_operation Rbind::ROperation.new(self.name)
        add_operation Rbind::ROperation.new(self.name,nil,Rbind::RParameter.new("other",self))

        paras = 0.upto(size-1).map do |idx|
            Rbind::RParameter.new("t#{idx}",type)
        end
        add_operation Rbind::ROperation.new(self.name,nil,paras)
        add_operation Rbind::ROperation.new("all",self,Rbind::RParameter.new("alpha",type))
        add_operation Rbind::ROperation.new("mul",self,Rbind::RParameter.new("other",self))
        add_operation Rbind::ROperation.new("conj",self) if size == 2 && !type.name =~/int/
        add_operation Rbind::ROperation.new("operator==",type.owner.bool,Rbind::RParameter.new("vec",self))
        add_operation Rbind::ROperation.new("operator!=",type.owner.bool,Rbind::RParameter.new("vec",self))
        add_operation Rbind::ROperation.new("operator+",self,Rbind::RParameter.new("vec",self))
        add_operation Rbind::ROperation.new("operator-",self,Rbind::RParameter.new("vec",self))
        add_operation Rbind::ROperation.new("operator*",self,Rbind::RParameter.new("vec",type))
        add_operation Rbind::ROperation.new("operator/",self,Rbind::RParameter.new("vec",type))
    end
end
