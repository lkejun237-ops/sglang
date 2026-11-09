// Exercise production helpers and the actual build shell; only external transport is stubbed.
def file = new File(args[0])
def source = file.text
def helperSource = source.split(/(?m)^pipeline \{/)[0]
def sha = 'a' * 40
def oldSha = 'b' * 40
def repo = '829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime'
def env = [BUILD_TARGET:'release', GIT_SHA:sha, ECR_IMAGE_REPOSITORY:repo,
           DOCKERFILE_PATH:'docker/whitebox.Dockerfile', PYTHON_BASE_IMAGE:'python@sha256:' + 'c'*64,
           GPU_BASE_IMAGE:'gpu@sha256:' + 'd'*64]
def files = ['docker/whitebox.Dockerfile':'FROM python:3.14 AS base\nFROM base AS release\n',
             'docker/whitebox.Dockerfile.dockerignore':'python/sglang/**\n',
             'benchmark/whitebox/requirements.txt':'pytest==9\n',
             'benchmark/whitebox/sdk-revision.txt':'e99e03be52410a37732e2007969de862ecfb75b5\n']
def hashWork = File.createTempDir('whitebox-recipe-hasher-', '')
try {
def hashFiles = []
def hashFailure = false
def invalidHash = false
def writeFailure = false
def hashDirectory = null
def binding = new Binding([env:env, params:[BRANCH_NAME:'main'],
    readFile:{ arg -> arg instanceof Map ? files[arg.file].bytes.encodeBase64().toString() : files[arg] },
    fileExists:{ path -> files.containsKey(path) },
    error:{ msg -> throw new IllegalArgumentException(msg.toString()) }, echo:{ msg -> }])
binding.setVariable('pwd',{ Map options -> assert options.tmp;hashWork.path })
binding.setVariable('dir',{ String path, Closure body -> hashDirectory=new File(path);hashDirectory.mkdirs();body() })
binding.setVariable('writeFile',{ Map options ->
    if(writeFailure)throw new IllegalStateException('write failed')
    assert options.encoding == 'UTF-8'
    def output = new File(hashDirectory,options.file)
    output.setText(options.text,'UTF-8');hashFiles << output.bytes
})
binding.setVariable('container',{ name, Closure body -> assert name=='aws';body() })
binding.setVariable('sh',{ Map options ->
    if(hashFailure)throw new IllegalStateException('hasher failed')
    if(invalidHash)return 'invalid checksum output'
    def p=new ProcessBuilder('sh','-c',options.script).directory(hashDirectory).redirectErrorStream(true).start()
    def output=p.inputStream.text;assert p.waitFor()==0:output
    return output
})
binding.setVariable('deleteDir',{ assert hashDirectory.name=='whitebox-layer-recipe';hashDirectory.deleteDir() })
def script = new GroovyShell(binding).parse(helperSource)
script.prepareWhiteboxCacheRecipe()
def recipe = env.WHITEBOX_CACHE_RECIPE
assert hashFiles: 'The recipe must hash a real canonical file through the pipeline shell'
assert !new File(hashWork,'whitebox-layer-recipe').exists()

assert recipe ==~ /[a-f0-9]{64}/
// Actual managed bytes, rather than repository defaults, control identity.
files['docker/whitebox.Dockerfile'] += '# managed recipe update\n'
script.prepareWhiteboxCacheRecipe(); assert env.WHITEBOX_CACHE_RECIPE != recipe
files['docker/whitebox.Dockerfile'] = 'FROM python:3.14 AS base\nFROM base AS release\n'
script.prepareWhiteboxCacheRecipe(); assert env.WHITEBOX_CACHE_RECIPE == recipe
// Capture actual managed ARG/base and syntax identities, effective ignore,
// requirements and the Room-only pin through the production preparation helper.
def baselineFiles = new LinkedHashMap(files)
['docker/whitebox.Dockerfile.dockerignore','benchmark/whitebox/requirements.txt'].each { path ->
    files[path] += 'changed\n';script.prepareWhiteboxCacheRecipe();assert env.WHITEBOX_CACHE_RECIPE != recipe
    files.clear();files.putAll(baselineFiles)
}
env.GIT_SHA=oldSha;script.prepareWhiteboxCacheRecipe();assert env.WHITEBOX_CACHE_RECIPE == recipe;env.GIT_SHA=sha
files['benchmark/whitebox/sdk-revision.txt']=oldSha
script.prepareWhiteboxCacheRecipe();assert env.WHITEBOX_CACHE_RECIPE == recipe
files[env.DOCKERFILE_PATH]='ARG PYTHON_IMAGE=default:tag\nFROM ${PYTHON_IMAGE} AS base\nFROM base AS release\n'
assert script.whiteboxEffectiveBaseImages(files[env.DOCKERFILE_PATH],[PYTHON_IMAGE:'selected:tag']) == ['selected:tag']
assert script.whiteboxEffectiveBaseImages('FROM one:tag AS base\nFROM base AS release\nFROM two:tag AS room\n',[:]) == ['one:tag','two:tag']
script.prepareWhiteboxCacheRecipe();def argRecipe=env.WHITEBOX_CACHE_RECIPE
env.PYTHON_BASE_IMAGE='another:base';script.prepareWhiteboxCacheRecipe();assert env.WHITEBOX_CACHE_RECIPE != argRecipe
files.clear();files.putAll(baselineFiles)
env.BUILD_TARGET='room';script.prepareWhiteboxCacheRecipe();def roomRecipe=env.WHITEBOX_CACHE_RECIPE
files['benchmark/whitebox/sdk-revision.txt']=oldSha
script.prepareWhiteboxCacheRecipe();assert env.WHITEBOX_CACHE_RECIPE != roomRecipe
files.clear();files.putAll(baselineFiles);env.BUILD_TARGET='release';script.prepareWhiteboxCacheRecipe()
def inputs = [dockerfileBase64: files[env.DOCKERFILE_PATH].bytes.encodeBase64().toString(),
    ignoreBase64:'aWdub3Jl', requirementsBase64:'cmVx', target:'release', platform:'linux/amd64',
    baseImages:['python:3.14'], buildkit:'moby/buildkit:v0.17.2-rootless', frontend:'dockerfile.v0', sdkRevision:oldSha]
def tiny = [dockerfileBase64:'Zg==',ignoreBase64:'',requirementsBase64:'cg==',target:'release',
    platform:'linux/amd64',baseImages:['python:tag'],buildkit:'buildkit:tag',frontend:'dockerfile.v0']
assert script.whiteboxCacheRecipe(tiny) == 'e651f30c0aa5c44b8a10af732bd0d217cc1e60ae14d6bf7c530fccbf43a6260f'
assert new String(hashFiles.last(),'UTF-8') == '{"schema":1,"dockerfileBase64":"Zg==","ignoreBase64":"","requirementsBase64":"cg==","target":"release","platform":"linux/amd64","baseImages":["python:tag"],"buildkit":"buildkit:tag","frontend":"dockerfile.v0","sdkRevision":""}'
def injectedFile=new File(hashWork,'injected')
def injection='$(touch '+injectedFile.path+'); `touch '+injectedFile.path+'`; "quoted"\n'
script.whiteboxCacheRecipe(tiny + [frontend:injection])
assert new groovy.json.JsonSlurperClassic().parseText(new String(hashFiles.last(),'UTF-8')).frontend == injection
assert !injectedFile.exists(): 'Canonical input must never execute as shell syntax'
['write','shell','output'].each { failure ->
    writeFailure=failure=='write';hashFailure=failure=='shell';invalidHash=failure=='output'
    boolean rejected=false
    try {script.whiteboxCacheRecipe(tiny)} catch(IllegalStateException | IllegalArgumentException expectedFailure) {rejected=true}
    assert rejected
    assert !new File(hashWork,'whitebox-layer-recipe').exists(): 'Failure must clean the owned canonical directory'
    writeFailure=false;hashFailure=false;invalidHash=false
}
def canonical = script.whiteboxCacheRecipe(inputs)
assert script.whiteboxCacheRecipe(inputs + [sourceRevision:sha, runtimeSource:'changed', testSource:'changed']) == canonical
['dockerfileBase64','ignoreBase64','requirementsBase64','platform','buildkit','frontend'].each { key ->
    assert script.whiteboxCacheRecipe(inputs + [(key):inputs[key] + 'changed']) != canonical
}
assert script.whiteboxCacheRecipe(inputs + [baseImages:['python:3.15']]) != canonical
assert script.whiteboxCacheRecipe(inputs + [sdkRevision:sha]) == canonical
assert script.whiteboxCacheRecipe(inputs + [target:'room',sdkRevision:sha]) != script.whiteboxCacheRecipe(inputs + [target:'room',sdkRevision:oldSha])
def expected = [repository:repo,branch:'main',target:'release',recipeSha256:recipe,platform:'linux/amd64']
def valid = expected + [digest:'sha256:'+'1'*64,sourceRevision:oldSha,integrityVerified:true]
assert script.selectWhiteboxCacheCandidates(expected,[valid]) == [repo+'@'+valid.digest]
['repository','branch','target','recipeSha256','platform'].each { key ->
    assert script.selectWhiteboxCacheCandidates(expected,[valid + [(key):'wrong']]).empty
}
assert script.selectWhiteboxCacheCandidates(expected,[valid + [sourceRevision:'short'],valid + [integrityVerified:false],[:]]).empty
assert script.selectWhiteboxCacheCandidates(expected,[valid,valid]).size() == 1
assert script.selectWhiteboxCacheCandidates(expected,(1..8).collect { valid + [digest:'sha256:'+it.toString()*64] }).size() == 3
assert script.selectWhiteboxCacheCandidates(expected,Collections.nCopies(20,[:]) + [valid]).empty
// Execute the real stage command against a fake buildctl; imports must reach argv.
def shell = source.substring(source.indexOf("stage('Build and Push Immutable Image')")).split("sh '''",2)[1].split("'''",2)[0]
def temp = File.createTempDir('whitebox-cache-', '')
try {
    new File(temp,'benchmark/whitebox').mkdirs()
    new File(temp,'benchmark/whitebox/sdk-revision.txt').text = files['benchmark/whitebox/sdk-revision.txt']
    def secrets = new File(temp,'secrets'); secrets.mkdirs()
    new File(secrets,'world-model-cache-imports').text = repo+'@'+valid.digest+'\n'
    def bin = new File(temp,'buildctl-daemonless.sh')
    bin.text = '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGV_FILE"\n'; bin.setExecutable(true)
    def command = new File(temp,'build.sh'); command.text = shell.replace('/build-secrets',secrets.path)
    def argsFile = new File(temp,'args')
    def proc = new ProcessBuilder('sh',command.path).directory(temp).redirectErrorStream(true)
    proc.environment().putAll([PATH:temp.path+':'+System.getenv('PATH'),ARGV_FILE:argsFile.path,
        WORKSPACE:temp.path,DOCKERFILE_PATH:env.DOCKERFILE_PATH,PYTHON_BASE_IMAGE:env.PYTHON_BASE_IMAGE,
        GPU_BASE_IMAGE:env.GPU_BASE_IMAGE,GIT_SHA:sha,BUILD_TARGET:'release',
        WHITEBOX_CACHE_RECIPE:recipe,WHITEBOX_CACHE_BRANCH:'main',WHITEBOX_SOURCE_BRANCH:'main',BUILDKIT_IMAGE_OUTPUT:'type=image,name=current,push=true'])
    def p=proc.start(); def out=p.inputStream.text; assert p.waitFor() == 0:out
    def argv=argsFile.readLines()
    assert argv.contains('type=registry,ref='+repo+'@'+valid.digest)
    assert argv.contains('build-arg:SOURCE_REVISION='+sha)
    assert argv.contains('build-arg:WHITEBOX_SOURCE_BRANCH=main')
    assert argv.contains('label:org.opencontainers.image.revision='+sha)
    assert argv.contains('label:org.worldkit.whitebox.cache.recipe-sha256='+recipe)
    assert argv.contains('type=inline')
    assert !argv.any { it.contains('buildcache-') || it.contains('loopit-build-cache') }
} finally { temp.deleteDir() }
assert !source.substring(source.indexOf('pipeline {')).contains('prepareSharedBuildCache(')
assert !source.substring(source.indexOf('pipeline {')).contains('configureSharedBuildCache(')
println 'cache recipe, selection and actual build command cases passed'

// Execute original registry helper -> imports append -> Declarative build/reuse
// stages, replacing only external registry transport and the buildctl executable.
def work = File.createTempDir('whitebox-cache-wiring-', '')
try {
    def secrets = new File(work,'secrets'); secrets.mkdirs()
    def imports = new File(secrets,'world-model-cache-imports')
    def chown = new File(work,'chown'); chown.text='#!/bin/sh\nexit 0\n';chown.setExecutable(true)
    env.AWS_REGION='us-east-2';env.ECR_REPOSITORY='leap-world/minwm-realtime';env.SAME_SHA_PRESENT='false'
    def fakeAws = new File(work,'aws')
    fakeAws.text = '#!/bin/sh\n[ "$SAME_SHA_PRESENT" = true ]\n'; fakeAws.setExecutable(true)
    def bin = new File(work,'buildctl-daemonless.sh')
    bin.text = '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGV_FILE"\n'; bin.setExecutable(true)
    def runShell = { String text ->
        def p = new ProcessBuilder('sh','-c',text.replace('/build-secrets',secrets.path)).directory(work).redirectErrorStream(true)
        p.environment().putAll(env.collectEntries { k,v -> [(k):v.toString()] })
        p.environment().putAll([PATH:work.path+':'+System.getenv('PATH'),ARGV_FILE:new File(work,'args').path,
            WORKSPACE:work.path,WORLD_MODEL_CACHE_TAG_PREFIXES:'whitebox-control',
            IMAGE_REPOSITORY:repo,BUILDKIT_IMAGE_OUTPUT:'type=image,name=current,push=true'])
        def process=p.start(); def output=process.inputStream.text
        assert process.waitFor()==0:output
    }
    def fixture = groovy.json.JsonOutput.toJson([valid + [tag:'moved-tag']])
    boolean optionalFailure = false
    String finalDigest = ''
    def observed = []
    binding.setVariable('container',{ name, Closure body -> body() })
    binding.setVariable('withEnv',{ List pairs, Closure body ->
        def previous = new LinkedHashMap(env)
        pairs.each { pair -> def bits=pair.split('=',2);env[bits[0]]=bits[1] }
        try { body() } finally { env.clear();env.putAll(previous) }
    })
    binding.setVariable('sh',{ arg ->
        if (arg instanceof Map) {
            if (arg.label == '校验 ECR 边界并生成 BuildKit 凭据') { return '' }
            if (arg.label == '检查 immutable image 是否已存在') { return finalDigest }
            if (arg.label == 'Verify bounded cross-commit inline cache candidates') {
                if (optionalFailure) { throw new IllegalStateException('optional transport failed') }
                return fixture
            }
            runShell(arg.script)
        } else { runShell(arg) }
        return ''
    })
    binding.setVariable('echo',{ msg -> observed << msg.toString() })
    def stages = source.substring(source.indexOf("        stage('Build and Push Immutable Image')"),source.indexOf("        stage('Resolve Registry Digest')"))
    boolean enabled=true
    binding.setVariable('stage',{ name, Closure body -> enabled=true; body() })
    binding.setVariable('when',{ Closure body -> body() })
    binding.setVariable('expression',{ Closure body -> enabled=body() })
    binding.setVariable('steps',{ Closure body -> if(enabled)body() })
    env.IMAGE_REF=repo+':new-source-tag';env.ECR_IMAGE_REF=env.IMAGE_REF
    env.THAILAND_ACR_ONLY='false';env.WHITEBOX_CACHE_RECIPE=recipe;env.WHITEBOX_CACHE_BRANCH='main'
    script.prepareWorldModelEcr();script.discoverWhiteboxCacheCandidates()
    assert imports.readLines()==[repo+'@'+valid.digest]
    new GroovyShell(binding).evaluate(stages)
    assert new File(work,'args').readLines().contains('build-arg:SOURCE_REVISION='+sha)
    assert new File(work,'args').readLines().contains('type=registry,ref='+repo+'@'+valid.digest)
    assert !imports.text.contains('moved-tag')
    env.SAME_SHA_PRESENT='true'
    script.prepareWorldModelEcr();script.discoverWhiteboxCacheCandidates()
    assert imports.readLines()==[repo+':whitebox-control-'+sha,repo+'@'+valid.digest]
    env.SAME_SHA_PRESENT='false'
    new File(work,'args').delete()
    finalDigest='sha256:'+'f'*64
    script.prepareWorldModelEcr();script.discoverWhiteboxCacheCandidates()
    new GroovyShell(binding).evaluate(stages)
    assert !new File(work,'args').exists()
    assert observed.any { it.contains('跳过重复 push') }
    finalDigest='';optionalFailure=true
    script.prepareWorldModelEcr();script.discoverWhiteboxCacheCandidates()
    assert imports.text.empty
    new GroovyShell(binding).evaluate(stages)
    assert new File(work,'args').exists(): 'Optional lookup failure must still execute a cold build'
} finally {work.deleteDir()}
println 'actual registry/import/build/reuse and optional cold-build wiring passed'

} finally { hashWork.deleteDir() }
