RUNDIR=	sscResolution sscResolution/resolution cuda/ c/

all: install

install:
	python3 setup.py install --user

uninstall:
	pip uninstall sscResolution

clean:
	rm -fr build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*.egg-info/; rm -rf $$j/__pycache__/; rm -rf $$j/*~; done

