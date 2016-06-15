var socket = io.connect('/uploader');

var Button = ReactBootstrap.Button;
var Popover = ReactBootstrap.Popover;

var VideoCanvas = React.createClass({
    propTypes: {
      width: React.PropTypes.number.isRequired,
      height: React.PropTypes.number.isRequired,
      captureInterval: React.PropTypes.number.isRequired,
      updateFrameSize: React.PropTypes.func.isRequired,
      handleClick: React.PropTypes.func,
    },
    componentDidMount() {
      this.v = ReactDOM.findDOMNode(this.refs.video);
      this.c = ReactDOM.findDOMNode(this.refs.canvas);
      this.ctx = this.c.getContext('2d');

      // initialize video
      var that = this;
      var p = navigator.mediaDevices.getUserMedia({audio: false, video: true});
      p.then(function(mediaStream) {
          that.v.src = window.URL.createObjectURL(mediaStream);
          that.v.onloadedmetadata = that.updateFrameSize;
          that.v.onresize = that.updateFrameSize;
          // start video capture
          that.play();
      });
    },
    updateFrameSize() {
      this.props.updateFrameSize(this.v.videoWidth, this.v.videoHeight);
    },
    drawShapes() {
      // overlay shapes
      var that = this;
      // circle
      this.props.shapes.forEach(function(shape_func) {
          shape_func(that.ctx);
      });
    },
    drawCanvas(src, src_region, dst_region) {
      // region check
      if (!src_region || (src_region.x < 0 || src_region.y < 0 ||
                          src_region.w <= 0 || src_region.h <= 0)) {
        src_region = {x: 0, y: 0, w: src.width, h: src.height};
      }
      if (!dst_region || (dst_region.x < 0 || dst_region.y < 0 ||
                          dst_region.w <= 0 || dst_region.h <= 0)) {
        dst_region = {x: 0, y: 0, w: this.props.width, h: this.props.height};
      }

      // clear
      // this.ctx.clearRect(0, 0, this.props.width, this.props.height);

      // draw video frame
      this.ctx.drawImage(src,
        src_region.x, src_region.y, src_region.w, src_region.h,
        dst_region.x, dst_region.y, dst_region.w, dst_region.h);

      // draw shapes
      if (this.props.shapeDrawable) this.drawShapes();
    },
    drawVideo(region) {
      this.drawCanvas(this.v, region, region);
    },
    play() {
      this.stop(); // escape double loop

      // draw loop function
      var that = this;
      var drawLoop = function() {
        // draw
        that.drawVideo();
        // recursive call
        that.loopId = setTimeout(drawLoop, that.props.captureInterval);
      };
      drawLoop();
    },
    stop() {
      if (this.loopId) {
        clearTimeout(this.loopId);
        this.loopId = null;
      }
    },
    handleClick(e) {
      if (this.props.handleClick) this.props.handleClick(e);
    },
    render() {
      return (
        <div>
          <video ref="video"
           style={{display: "none"}}
           width={this.props.width}
           height={this.props.height}
           autoPlay="1" />
          <canvas ref="canvas"
           className="img-responsive"
           style={{backgroundColor: 'black'}}
           width={this.props.width}
           height={this.props.height} 
           onClick={this.handleClick} />
        </div>
      );
    }
});


var VideoUI = React.createClass({
    rectColor: '#00FF00',
    rectLlineWidth: 6,
    region: {x: -1, y: -1, w: -1, h: -1},
    propTypes: {
      videoWidth: React.PropTypes.number.isRequired,
      videoHeight: React.PropTypes.number.isRequired,
      onReset: React.PropTypes.func,
    },
    getInitialState() {
      return {
        uploadable: true,
        captureInterval: 30,
        shapeDrawable: true,
        rectangles: [],
        circles: [],
      };
    },
    componentDidMount() {
    },
    uploadImg() {
      // stop video
      this.setState({uploadable: false});
      this.refs.videocanvas.stop();
      // disable shape drawing
      var that = this;
      this.setState({shapeDrawable: false}, function() {
          // draw canvas with no shapes
          that.refs.videocanvas.drawVideo();
          // get image data
          var data = that.refs.videocanvas.c.toDataURL('image/png');
          // emit
          socket.emit('upload_img', {img: data, region: that.region});
          // draw shapes for usability
          that.refs.videocanvas.drawShapes();
      });
    },
    onResult(data) {
      // check whether this request is canceled
      if (this.state.uploadable) return;

      // draw received image
      var that = this;
      var img = new Image();
      img.src = data.img;
      img.onload = () => {
        that.refs.videocanvas.stop();
        if (data.img_options && data.img_options.region) {
          that.refs.videocanvas.drawCanvas(img, null, that.region);
          that.refs.videocanvas.drawShapes();
        } else {
          that.refs.videocanvas.drawCanvas(img, null, null);
        }
      };
    },
    onReset() {
      // call parent event
      if (this.props.onReset) this.props.onReset();
      // clear region
      this.region = {x: -1, y: -1, w: -1, h: -1};
      this.setState({rectangles: []});
      this.setState({circles: []});
      // restart video
      this.setState({uploadable: true});
      this.setState({shapeDrawable: true});
      this.refs.videocanvas.play();
    },
    onVideoSizeChanged(width, height) {
      // clear region
      this.onReset();
      // set component size
      this.props.onVideoSizeChanged(width, height);
    },
    handleClick(e) {
      // region editing
      var brect = e.target.getBoundingClientRect();
      var clickX = (e.clientX - brect.left);
      var clickY = (e.clientY - brect.top);
      // normalize
      clickX = clickX / brect.width * this.props.videoWidth;
      clickY = clickY / brect.height * this.props.videoHeight;

      // check click range
      if (clickX < 0 || this.props.videoWidth <= clickX ||
          clickY < 0 || this.props.videoHeight <= clickY) return;

      // switch point mode
      var circle = this.newCircle(clickX, clickY);
      if (this.region.x < 0 || this.region.y < 0 ||
          this.region.w > 0 || this.region.h > 0) {
        // escape invisible point
        this.onReset();
        // first point
        this.region.x = clickX;
        this.region.y = clickY;
        // clear previous region
        this.region.w = -1;
        this.region.h = -1;
        // set drawing
        this.setState({rectangles: []});
        this.setState({circles: [circle]});
      } else {
        // second point
        this.region.w = Math.abs(clickX - this.region.x);
        this.region.h = Math.abs(clickY - this.region.y);
        // escape negative region size
        this.region.x = Math.min(this.region.x, clickX);
        this.region.y = Math.min(this.region.y, clickY);
        // set drawing
        var rect = this.newRect(this.region.x, this.region.y,
                                this.region.w, this.region.h);
        this.setState({rectangles: [rect]});
        this.setState({circles: this.state.circles.concat([circle])});
      }
    },
    newRect(x, y, w, h) {
      var that = this;
      return (function(rect) {
          return function(ctx) {
            ctx.strokeStyle = that.rectColor;
            ctx.lineWidth = that.rectLlineWidth;
            ctx.beginPath();
            ctx.rect(rect.x, rect.y, rect.w, rect.h);
            ctx.stroke();
          };
      })({x: x, y: y, w: w, h: h});
    },
    newCircle(x, y) {
      var that = this;
      return (function(circle) {
          return function(ctx) {
            ctx.fillStyle = that.rectColor;
            ctx.strokeStyle = that.rectColor;
            ctx.beginPath();
            ctx.arc(circle.x, circle.y, circle.r, 0, 2 * Math.PI, false);
            ctx.fill();
            ctx.stroke();
          };
      })({x: x, y: y, r: this.rectLlineWidth});
    },
    render() {
      return (
        <div>
          <VideoCanvas ref="videocanvas" 
           width={this.props.videoWidth} height={this.props.videoHeight}
           captureInterval={this.state.captureInterval}
           updateFrameSize={this.onVideoSizeChanged}
           handleClick={this.handleClick} 
           shapeDrawable={this.state.shapeDrawable}
           shapes={this.state.circles.concat(this.state.rectangles)} />
          <div style={{maxWidth: this.props.videoWidth}}>
            <Button bsStyle="primary" bsSize="large"
             className="col-xs-6"
             disabled={this.state.uploadable ? false : true}
             onClick={this.uploadImg}>Upload</Button>
            <Button bsStyle="default" bsSize="large"
             className="col-xs-6"
             onClick={this.onReset}>Reset</Button>
          </div>
        </div>
      );
    }
});


var Message = React.createClass({
    propTypes: {
      width: React.PropTypes.number.isRequired,
    },
    getInitialState() {
      return {
        message: '',
      };
    },
    componentDidMount() {
    },
    setMessage(message) {
      this.setState({message: message});
    },
    render() {
      return (
        <div className="col-xs-12"
         style={{maxWidth: this.props.width}}>
          {(() => {
            if (this.state.message) {
              return (
                <Popover id={0} placement="bottom"
                 style={{maxWidth: this.props.width}}>
                  {this.state.message}
                </Popover>
              );
            }
          })()}
        </div>
      );
    }
});


var MainView = React.createClass({
    getInitialState() {
      return {
        videoWidth: 400,
        videoHeight: 300,
      };
    },
    componentDidMount() {
      // socket.io event
      socket.on('connect', this.onConnect);
      socket.on('response', this.onResponse);
    },
    onConnect() {
    },
    onResponse(data) {
      if (data.img) this.refs.videoui.onResult(data);
      if (data.msg) this.refs.message.setMessage(data.msg);
    },
    onReset() {
      this.refs.message.setMessage('');
    },
    onVideoSizeChanged(width, height) {
      this.setState({videoWidth: width});
      this.setState({videoHeight: height});
    },
    render() {
      return (
        <div className="center-block">
          <div className="container">
            <div className="row">
              <div className="col-xs-12">
                <VideoUI ref="videoui"
                 videoWidth={this.state.videoWidth}
                 videoHeight={this.state.videoHeight}
                 onReset={this.onReset}
                 onVideoSizeChanged={this.onVideoSizeChanged} />
              </div>
              <div className="col-xs-12">
                <Message ref="message"
                 width={this.state.videoWidth} />
              </div>
            </div>
          </div>
        </div>
      );
    }
});


ReactDOM.render(
  <MainView />,
  document.getElementById('content')
);
