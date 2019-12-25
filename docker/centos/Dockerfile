FROM centos:7

RUN yum install -y centos-release-scl
RUN yum install -y devtoolset-8
SHELL [ "scl", "enable", "devtoolset-8" ]

RUN yum install -y git wget 
RUN yum install -y libtool clang-format-6.0

ADD https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo /etc/yum.repos.d
RUN yum install -y bazel
RUN wget https://github.com/bazelbuild/buildtools/releases/download/0.29.0/buildifier
RUN chmod +x buildifier
RUN mv buildifier /usr/bin

RUN yum install -y epel-release
RUN yum install -y python python-dev python-pip
RUN python -m pip install -U pip 
RUN python -m pip install -U future
RUN python -m pip install -U tensorflow 

ENV BAZEL_LINKOPTS=-static-libstdc++
ENV BAZEL_LINKLIBS=-l%:libstdc++.a